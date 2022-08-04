/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <sys/time.h>

#include "allreduce_client.h"

struct allreduce_metrics {
	double runtime_min;		/* Minimum time of doing allreduce batch, in seconds */
	double runtime_max;		/* Maximum time of doing allreduce batch, in seconds */
	double runtime_avg;		/* Average time of doing allreduce batch, in seconds */
	double runtime_total;		/* Total time of doing all allreduce batches, in seconds */
	size_t current_batch_iter;	/* Current batch iteration number */
	const char *mode_str;		/* Pointer to the string of allreduce mode */
	const char *datatype_str;	/* Pointer to the string of allreduce datatype */
	const char *operation_str;	/* Pointer to the string of allreduce operation */
	size_t batch_size;		/* Allreduce operations in a single batch */
	size_t vector_size;		/* Allreduce vector size */
	int compute_repeats;		/* Indicates how many compute repetitions should be done to make "computation
					 * time" equal to "pure network time"
					 */
	double compute_time;		/* Pure computation time*/
	double network_time;		/* Pure network time */
	double overlap;			/* Percentage of overlap between computation and network operations */
};

typedef void (*allreduce_submit_func)(struct allreduce_super_request *allreduce_super_request);

static void **allreduce_vectors;	/* Array of allreduce vectors */
static size_t allreduce_next_id;	/* Next allreduce ID which could be allocated by clients */

DOCA_LOG_REGISTER(ALLREDUCE::Client);


static void
allreduce_vectors_cleanup(size_t num_allreduce_vectors)
{
	size_t vector_iter;

	/* Go through all vectors and free the memory allocated to hold them */
	for (vector_iter = 0; vector_iter < num_allreduce_vectors; ++vector_iter)
		free(allreduce_vectors[vector_iter]);

	free(allreduce_vectors);
}

static void
allreduce_vectors_reset(void)
{
	size_t vector_iter, i;

	/* Go through all vectors, fill them by initial data */
	for (vector_iter = 0; vector_iter < allreduce_config.batch_size; ++vector_iter) {
		for (i = 0; i < allreduce_config.vector_size; ++i) {
			/* Initialize vectors by initial values */
			switch (allreduce_config.datatype) {
			case ALLREDUCE_BYTE:
				((uint8_t *)allreduce_vectors[vector_iter])[i] = i % UINT8_MAX;
				break;
			case ALLREDUCE_INT:
				((int *)allreduce_vectors[vector_iter])[i] = i % INT_MAX;
				break;
			case ALLREDUCE_FLOAT:
				((float *)allreduce_vectors[vector_iter])[i] = (float)i;
				break;
			case ALLREDUCE_DOUBLE:
				((double *)allreduce_vectors[vector_iter])[i] = (double)i;
				break;
			}
		}
	}
}

static int
allreduce_vectors_init(void)
{
	size_t vector_size = allreduce_config.vector_size * allreduce_datatype_size[allreduce_config.datatype];
	size_t vector_iter, num_allreduce_vectors = 0;

	allreduce_vectors = malloc(allreduce_config.batch_size * sizeof(*allreduce_vectors));
	if (allreduce_vectors == NULL) {
		DOCA_LOG_ERR("failed to allocate memory to hold array of allreduce vectors");
		return -1;
	}

	/* Go through all vectors, allocate them */
	for (vector_iter = 0; vector_iter < allreduce_config.batch_size; ++vector_iter) {
		allreduce_vectors[vector_iter] = malloc(vector_size);
		if (allreduce_vectors[vector_iter] != NULL)
			++num_allreduce_vectors;
		else {
			allreduce_vectors_cleanup(num_allreduce_vectors);
			return -1;
		}
	}

	return 0;
}

/* Allocate vector for send operation and move iterator to the next one which will be allocated next time */
static inline void *
preallocated_vector_get(void)
{
	static size_t allreduce_vector_iter;
	void *vector = allreduce_vectors[allreduce_vector_iter];

	allreduce_vector_iter = (allreduce_vector_iter + 1) % allreduce_config.batch_size;
	return vector;
}

/* Client callback to complete AM send operation for receiving the allreduce result from the daemon */
static void
client_am_recv_data_complete_callback(void *arg, ucs_status_t status)
{
	struct allreduce_super_request *allreduce_super_request = arg;

	assert(status == UCS_OK);

	g_hash_table_remove(allreduce_super_requests_hash, &allreduce_super_request->header.id);
}

/* AM receive callback which is invoked when the client gets completion of the whole allreduce operation from daemon */
int
client_am_recv_ctrl_callback(struct allreduce_ucx_am_desc *am_desc)
{
	struct allreduce_ucx_connection *connection;
	const struct allreduce_header *allreduce_header;
	struct allreduce_super_request *allreduce_super_request;
	size_t header_length, length;

	allreduce_ucx_am_desc_query(am_desc, &connection, (const void **)&allreduce_header, &header_length,
						&length);

	assert(sizeof(*allreduce_header) == header_length);

	allreduce_super_request = g_hash_table_lookup(allreduce_super_requests_hash, &allreduce_header->id);
	if (allreduce_super_request == NULL) {
		DOCA_LOG_ERR("failed to find allreduce request with id=%zu in hash", allreduce_header->id);
		return -1;
	}

	/* Continue receiving data to the allocated vector */
	allreduce_ucx_am_recv(am_desc, allreduce_super_request->result_vector, length,
					client_am_recv_data_complete_callback, allreduce_super_request, NULL);

	return 0;
}

/* Set default values to the fields of the allreduce metrics */
static void
allreduce_metrics_reset(struct allreduce_metrics *allreduce_metrics)
{
	allreduce_metrics->runtime_min = DBL_MAX;
	allreduce_metrics->runtime_max = 0.;
	allreduce_metrics->runtime_total = 0.;
	allreduce_metrics->runtime_avg = 0.;
	allreduce_metrics->current_batch_iter = 0;
	allreduce_metrics->mode_str = allreduce_mode_str[allreduce_config.allreduce_mode];
	allreduce_metrics->datatype_str = allreduce_datatype_str[allreduce_config.datatype];
	allreduce_metrics->operation_str = allreduce_operation_str[allreduce_config.operation];
	allreduce_metrics->batch_size = allreduce_config.batch_size;
	allreduce_metrics->vector_size = allreduce_config.vector_size;
	allreduce_metrics->compute_repeats = 0;
	allreduce_metrics->compute_time = -1.;
	allreduce_metrics->network_time = -1.;
	allreduce_metrics->overlap = -1.;
}

/* Calculate allreduce metrics after allreduce batch successfully done */
static void
allreduce_metrics_calculate(double run_time, double compute_time,
				struct allreduce_metrics *allreduce_metrics)
{
	double overlapped_time, max_possible_overlapped_time;

	/* Minimum run time */
	if (run_time < allreduce_metrics->runtime_min)
		allreduce_metrics->runtime_min = run_time;

	/* Maximum run time */
	if (run_time > allreduce_metrics->runtime_max)
		allreduce_metrics->runtime_max = run_time;

	/* Total run time */
	allreduce_metrics->runtime_total += run_time;

	/* Average run time */
	allreduce_metrics->runtime_avg =
			allreduce_metrics->runtime_total / (allreduce_metrics->current_batch_iter + 1);

	/*
	 * Overlapped time is a difference between "pure network time" + "current compute time" and
	 * "current average run time"
	 */
	overlapped_time = allreduce_metrics->network_time + compute_time - allreduce_metrics->runtime_avg;

	/* Maximum possible overlapped time is a maximum between "pure network time" and "current compute time" */
	max_possible_overlapped_time = MAX(allreduce_metrics->network_time, compute_time);

	/*
	 * Percentage of computation/communication overlap calculated as:
	 * overlap = 100% * (overlapped_time / max_possible_overlapped_time), where
	 * (overlapped_time / max_possible_overlapped_time) should be in [0..1] range
	 */
	allreduce_metrics->overlap = 100. * MAX(0., MIN(1., overlapped_time / max_possible_overlapped_time));
}

static inline double
get_time(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return tv.tv_sec + (tv.tv_usec * 1e-6);
}

static void
matrix_multiplication(float **a, float *x, float *y, int size, int target_reps)
{
	int repeat, i, j;

	/* Do 'target_reps' iterations of matrix multiplication */
	for (repeat = 0; repeat < target_reps; repeat++) {
		/* Matrix multiplication */
		for (i = 0; i < size; ++i) {
			for (j = 0; j < size; ++j)
				x[i] += a[i][j] * y[j];
		}
	}
}

/* Calculate pure compute time */
static double
matrix_multiplication_avg_compute_time(float **a, float *x, float *y, int size, int num_reps)
{
	static const int discover_time_repeats = 10;
	double start_time, end_time;
	int repeat;

	/* Do 'discover_time_repeats' iterations of matrix multiplication */
	start_time = get_time();
	for (repeat = 0; repeat < discover_time_repeats; ++repeat)
		matrix_multiplication(a, x, y, size, num_reps);

	end_time = get_time();

	/* Calculate average compute time */
	return (end_time - start_time) / discover_time_repeats;
}

static void
cpu_exploit(struct allreduce_metrics *allreduce_metrics)
{
	/*
	 * Small matrix is used because the network time is could be small and we want to have a computation time close to
	 * the network time
	 */
	static const int size = 10;
	float **a, *x, *y;
	double start_time, end_time, estimated_overall_time;
	int num_reps;
	int i, j;

	assert(allreduce_metrics->network_time >= 0.);

	if (allreduce_metrics->network_time == 0.)
		return;

	/* Allocate memory for matrices */
	a = alloca(size * sizeof(*a));
	for (i = 0; i < size; ++i)
		a[i] = alloca(size * sizeof(**a));

	x = alloca(size * sizeof(*x));
	y = alloca(size * sizeof(*y));

	/* Allreduce metrics weren't initialized for computation */
	/* Initialize matrices */
	for (i = 0; i < size; ++i) {
		x[i] = 0.;
		y[i] = (float)i;
		for (j = 0; j < size; ++j)
			a[i][j] = (float)(i + j);
	}

	if (allreduce_metrics->compute_time < 0.) {
		/* Set some initial number of repetitions of matrix multiplications */
		num_reps = (50000000 / (2 * size * size)) + 1;
		/* Calculate average computation time of doing 'num_reps' repetitions */
		estimated_overall_time = matrix_multiplication_avg_compute_time(a, x, y, size, num_reps);

		/* Calculate repetitions of computations to be approximately equal to 'network_time' */
		allreduce_metrics->compute_repeats = MAX(1, (int)((num_reps * allreduce_metrics->network_time) /
									estimated_overall_time));

		/* Calculate computation time took by calculated 'compute_repeats' iterations of matrix multiplications */
		start_time = get_time();
		matrix_multiplication(a, x, y, size, allreduce_metrics->compute_repeats);
		end_time = get_time();
		allreduce_metrics->compute_time = end_time - start_time;
	} else {
		/* Do matrix multiplication */
		matrix_multiplication(a, x, y, size, allreduce_metrics->compute_repeats);
	}
}

static void
allreduce_metrics_iteration_print(double run_time, double compute_time, struct allreduce_metrics *allreduce_metrics)
{
	DOCA_LOG_INFO("%zu: current run time - %.3f seconds, compute - %.3f seconds, min - %.3f seconds, max - %.3f seconds, avg - %.3f seconds",
		allreduce_metrics->current_batch_iter, run_time, compute_time, allreduce_metrics->runtime_min,
		allreduce_metrics->runtime_max, allreduce_metrics->runtime_avg);
}

static void
allreduce_metrics_print(struct allreduce_metrics *allreduce_metrics)
{
	DOCA_LOG_INFO("Allreduce (%s/%s/%s) and matrix multiplication metrics to complete %zu batches (batch size - %zu, vector size - %zu):",
			allreduce_metrics->mode_str, allreduce_metrics->operation_str, allreduce_metrics->datatype_str,
			allreduce_metrics->current_batch_iter, allreduce_metrics->batch_size,
			allreduce_metrics->vector_size);
	DOCA_LOG_INFO("min - %.3f seconds", allreduce_metrics->runtime_min);
	DOCA_LOG_INFO("max - %.3f seconds", allreduce_metrics->runtime_max);
	DOCA_LOG_INFO("avg - %.3f seconds", allreduce_metrics->runtime_avg);
	DOCA_LOG_INFO("total - %.3f seconds", allreduce_metrics->runtime_total);
	DOCA_LOG_INFO("computation time - %.3f seconds", allreduce_metrics->compute_time);
	DOCA_LOG_INFO("pure network time - %.3f seconds", allreduce_metrics->network_time);
	DOCA_LOG_INFO("computation and communication overlap - %.2f%%", allreduce_metrics->overlap);
}

static void
allreduce_offloaded_complete_ctrl_send_callback(void *arg, ucs_status_t status)
{
	assert(status == UCS_OK);
}

static void
allreduce_offloaded_submit(struct allreduce_super_request *allreduce_super_request)
{
	struct allreduce_ucx_connection *connection = connections[0];

	/* Send allreduce control Active Message with allreduce data to the daemon for futher processing */
	allreduce_ucx_am_send(connection, ALLREDUCE_CTRL_AM_ID, &allreduce_super_request->header,
					sizeof(allreduce_super_request->header),
					allreduce_super_request->result_vector,
					allreduce_super_request->result_vector_size *
					allreduce_datatype_size[allreduce_config.datatype],
					allreduce_offloaded_complete_ctrl_send_callback, NULL, NULL);
}

static void
allreduce_non_offloaded_submit(struct allreduce_super_request *allreduce_super_request)
{
	allreduce_scatter(allreduce_super_request); /* Do allreduce operation among other clients */
}

/* Submit a batch of offloaded allreduce operations */
static size_t
allreduce_batch_submit(size_t vector_size, size_t batch_size, allreduce_submit_func submit_func)
{
	struct allreduce_header allreduce_header;
	struct allreduce_super_request *allreduce_super_request;
	size_t op_iter;

	assert((vector_size == 0) || (vector_size == allreduce_config.vector_size));

	/* Go over all required allreduce requests */
	for (op_iter = 0; op_iter < batch_size; ++op_iter) {
		allreduce_header.id = allreduce_next_id++;

		allreduce_super_request = allreduce_super_request_get(&allreduce_header, vector_size,
									preallocated_vector_get());
		if (allreduce_super_request != NULL)
			submit_func(allreduce_super_request);
	}

	return op_iter;
}

static void
allreduce_batch_wait(void)
{
	/* Wait for completions of all submitted allreduce operations */
	while (g_hash_table_size(allreduce_super_requests_hash) > 0)
		allreduce_ucx_progress(context);
}

static void
allreduce_barrier(allreduce_submit_func submit_func)
{
	/* Do 0-byte allreduce operation to make sure all clients and daemons are up and running */
	allreduce_batch_submit(0, 1, submit_func);
	allreduce_batch_wait();
}

static void
allreduce_metrics_init(struct allreduce_metrics *allreduce_metrics, allreduce_submit_func submit_func)
{
	static const int discover_time_repeats = 3;
	double start_time, end_time;
	int repeat;

	allreduce_metrics_reset(allreduce_metrics);

	/* Reset vectors by initial data prior submitting allreduce */
	allreduce_vectors_reset();

	/* Calculate a pure network time consumed by a single batch of allreduce operations */
	DOCA_LOG_DBG("Performing %d batches to calculate estimated network time per batch.", discover_time_repeats);
	start_time = get_time();
	for (repeat = 0; repeat < discover_time_repeats; ++repeat) {
		allreduce_batch_submit(allreduce_config.vector_size, allreduce_config.batch_size, submit_func);
		allreduce_batch_wait();
	}
	end_time = get_time();

	/* Calculate average pure network time */
	allreduce_metrics->network_time = (end_time - start_time) / discover_time_repeats;

	/* Calculate average pure computation time */
	cpu_exploit(allreduce_metrics);
}

static void
allreduce(allreduce_submit_func submit_func)
{
	struct allreduce_metrics allreduce_metrics;
	size_t batch_size = allreduce_config.batch_size;
	double start_time, end_time, run_time;
	double compute_start_time, compute_end_time, compute_time;

	/* Post a barrier to make sure all clients and daemons are up and running prior benchmarking to avoid imbalance */
	DOCA_LOG_INFO("Making sure all participants in the Allreduce operation are available...");
	allreduce_barrier(submit_func);

	allreduce_metrics_init(&allreduce_metrics, submit_func);

	DOCA_LOG_INFO("Starting Allreduce operation.\n");

	/* Do benchmarking of allreduce */
	for (allreduce_metrics.current_batch_iter = 0;
		allreduce_metrics.current_batch_iter < allreduce_config.num_batches;
		++allreduce_metrics.current_batch_iter) {
		/* Reset vectors by initial data prior submitting allreduce */
		allreduce_vectors_reset();

		/* Calculate time of run time for performing batch of allreduce operations and computation */
		start_time = get_time();
		allreduce_batch_submit(allreduce_config.vector_size, batch_size, submit_func);

		compute_start_time = get_time();
		cpu_exploit(&allreduce_metrics);
		compute_end_time = get_time();
		compute_time = compute_end_time - compute_start_time;
		allreduce_batch_wait();

		end_time = get_time();
		run_time = end_time - start_time;

		/* Calculate allreduce metrics and print metrics of the current iteration */
		allreduce_metrics_calculate(run_time, compute_time, &allreduce_metrics);
		allreduce_metrics_iteration_print(run_time, compute_time, &allreduce_metrics);
	}

	/* Print summary of allreduce benchmarking */
	allreduce_metrics_print(&allreduce_metrics);
}

void
client_run(void)
{
	int ret;

	/* Allocate and fill vectors which contain data to do allreduce for */
	ret = allreduce_vectors_init();
	if (ret < 0)
		return;

	/* Setup receive handler for AM control messages from daemon which carry the allreduce result */
	allreduce_ucx_am_set_recv_handler(context, ALLREDUCE_CTRL_AM_ID, client_am_recv_ctrl_callback);

	/* Perform allreduce operations */
	switch (allreduce_config.allreduce_mode) {
	case ALLREDUCE_OFFLOAD_MODE:
		allreduce(allreduce_offloaded_submit);
		break;
	case ALLREDUCE_NON_OFFLOADED_MODE:
		allreduce(allreduce_non_offloaded_submit);
		break;
	default:
		DOCA_LOG_ERR("unsupported allreduce mode: %d", allreduce_config.allreduce_mode);
	}

	/* Destroy allreduce vectors */
	allreduce_vectors_cleanup(allreduce_config.batch_size);
}
