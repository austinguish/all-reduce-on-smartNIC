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

#include <signal.h>
#include <errno.h>
#include <errno.h>

#include "allreduce_daemon.h"

static volatile int running = 1;	/* Indicates if the process still running or not, used by daemons */

DOCA_LOG_REGISTER(ALLREDUCE::Daemon);

/* Daemon callback to complete receiving the vector with the data from the client to do allreduce for */
static void
daemon_am_recv_data_complete_callback(void *arg, ucs_status_t status)
{
	struct allreduce_request *allreduce_request = arg;
	struct allreduce_super_request *allreduce_super_request;

	assert(status == UCS_OK);

	/* Received size of the vector must be <= the configured size by a user */
	assert(allreduce_request->vector_size <= allreduce_config.vector_size);

	/* Try to find or allocate allreduce super request to match the allreduce request which is currently received */
	allreduce_super_request = allreduce_super_request_get(&allreduce_request->header, allreduce_request->vector_size,
								NULL);
	if (allreduce_super_request == NULL) {
		allreduce_request_destroy(allreduce_request);
		return;
	}

	/* Attach the received allreduce request to the allreduce super request for futher processing */
	allreduce_request->allreduce_super_request = allreduce_super_request;
	++allreduce_super_request->num_allreduce_requests;
	STAILQ_INSERT_TAIL(&allreduce_super_request->allreduce_requests_list, allreduce_request, entry);

	/* Do operation using the received vector and save the result to the vector from the allreduce super request */
	allreduce_do_reduce(allreduce_super_request, allreduce_request->vector);
	free(allreduce_request->vector);
	allreduce_request->vector = NULL;

	/* The whole result will be sent to the other daemons when all vectors are received from clients */
	if (allreduce_super_request->num_allreduce_requests == allreduce_config.num_clients) {
		/*
		 * Daemons received the allreduce vectors from all clients - perform allreduce among other daemons
		 * (if any)
		 */
		allreduce_scatter(allreduce_super_request);
	} else if (allreduce_super_request->num_allreduce_requests > allreduce_config.num_clients) {
		DOCA_LOG_WARN("More vectors than clients were received for a single Allreduce operation. Ignoring vector of new client (considered duplicates) but including in the final result response.");
	} else {
		/* Not all clients sent their vectors to the daemon */
	}
}

/* AM receive callback which is invoked when the daemon receives notification from some client */
int
daemon_am_recv_ctrl_callback(struct allreduce_ucx_am_desc *am_desc)
{
	struct allreduce_ucx_connection *connection;
	const struct allreduce_header *allreduce_header;
	struct allreduce_request *allreduce_request;
	size_t header_length, length;

	allreduce_ucx_am_desc_query(am_desc, &connection, (const void **)&allreduce_header, &header_length,
						&length);

	assert(sizeof(*allreduce_header) == header_length);

	/* Allreduce request will be freed in "daemon_allreduce_complete_client_operation_callback" function upon
	 * completion of sending results to all clients
	 */
	allreduce_request = allreduce_request_allocate(connection, allreduce_header, length);
	if (allreduce_request == NULL)
		return -1;

	/* Continue receiving data to the allocated vector */
	DOCA_LOG_DBG("Received a vector from a client.");
	allreduce_ucx_am_recv(am_desc, allreduce_request->vector, length, daemon_am_recv_data_complete_callback,
			      allreduce_request, NULL);

	return 0;
}

static void
signal_terminate_handler(int signo)
{
	running = 0;
}

static void
signal_terminate_set(void)
{
	struct sigaction new_sigaction = {
		.sa_handler = signal_terminate_handler,
		.sa_flags = 0
	};

	sigemptyset(&new_sigaction.sa_mask);

	if (sigaction(SIGINT, &new_sigaction, NULL) != 0) {
		DOCA_LOG_ERR("failed to set SIGINT signal handler: %s", strerror(errno));
		abort();
	}
}

void
daemon_run(void)
{
	if (allreduce_config.num_clients == 0) {
		/* Nothing to do */
		DOCA_LOG_INFO("Stop running - daemon doesn't have clients");
		return;
	}

	signal_terminate_set();

	/* Setup receive handler for AM control messages from client which carries a vector to do allreduce for */
	allreduce_ucx_am_set_recv_handler(context, ALLREDUCE_CTRL_AM_ID, daemon_am_recv_ctrl_callback);

	DOCA_LOG_INFO("Daemon is active and waiting for client connections... Press ctrl+C to terminate.");

	while (running) {
		/* Progress UCX to handle client's allreduce requests until signanl isn't received */
		allreduce_ucx_progress(context);
	}
}
