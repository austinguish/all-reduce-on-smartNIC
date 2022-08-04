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
#include <string.h>
#include <sys/time.h>

#include <utils.h>

#include "allreduce_core.h"

DOCA_LOG_REGISTER(ALLREDUCE::Core);

/* Names of allreduce process modes */
const char * const allreduce_role_str[] = {
	[ALLREDUCE_CLIENT] = "client",	/* Name of client allreduce role */
	[ALLREDUCE_DAEMON] = "daemon"	/* Name of daemon allreduce role */
};
/* Names of allreduce algorithms */
const char * const allreduce_mode_str[] = {
	[ALLREDUCE_NON_OFFLOADED_MODE] = "non-offloaded",	/* Name of non-offloaded allreduce algorithm */
	[ALLREDUCE_OFFLOAD_MODE] = "offloaded"			/* Name of offloaded allreduce algorithm */
};
const char * const allreduce_datatype_str[] = {
	[ALLREDUCE_BYTE] = "byte",	/* Name of "byte" datatype */
	[ALLREDUCE_INT] = "int",	/* Name of "int" datatype */
	[ALLREDUCE_FLOAT] = "float",	/* Name of "float" datatype */
	[ALLREDUCE_DOUBLE] = "double"	/* Name of "double" datatype */
};
const size_t allreduce_datatype_size[] = {
	[ALLREDUCE_BYTE] = sizeof(uint8_t),	/* Size of "byte" datatype */
	[ALLREDUCE_INT] = sizeof(int),		/* Size of "int" datatype */
	[ALLREDUCE_FLOAT] = sizeof(float),	/* Size of "float" datatype */
	[ALLREDUCE_DOUBLE] = sizeof(double)	/* Size of "double" datatype */
};
const char * const allreduce_operation_str[] = {
	[ALLREDUCE_SUM] = "sum",	/* Name of summation of two vector elements */
	[ALLREDUCE_PROD] = "prod"	/* Name of product of two vector elements */
};
struct allreduce_config allreduce_config = {0};	/* UCX allreduce configuration */
struct allreduce_ucx_context *context;		/* UCX context */
struct allreduce_ucx_connection **connections;	/* Array of UCX connections */
GHashTable *allreduce_super_requests_hash;	/* Hash which contains "ID -> allreduce super request" elements */

static void
set_role_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;
	const char *str = (const char *) param;

	if (strcmp(str, allreduce_role_str[ALLREDUCE_CLIENT]) == 0)
		app_config->role = ALLREDUCE_CLIENT;
	else if (strcmp(str, allreduce_role_str[ALLREDUCE_DAEMON]) == 0)
		app_config->role = ALLREDUCE_DAEMON;
	else
		APP_EXIT("unknow role '%s' was specified", str);
}

static void
set_allreduce_mode_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;
	const char *str = (const char *) param;

	if (strcmp(str, allreduce_mode_str[ALLREDUCE_OFFLOAD_MODE]) == 0)
		app_config->allreduce_mode = ALLREDUCE_OFFLOAD_MODE;
	else if (strcmp(str, allreduce_mode_str[ALLREDUCE_NON_OFFLOADED_MODE]) == 0)
		app_config->allreduce_mode = ALLREDUCE_NON_OFFLOADED_MODE;
	else
		APP_EXIT("unknow mode '%s' was specified", str);
}

static void
set_dest_ip_str_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->dest_addresses.str = strdup((char *) param);
	if (app_config->dest_addresses.str == NULL)
		APP_EXIT("failed to allocate memory to hold a list of destination addresses '%s'",
				(char *) param);
}

static void
set_dest_port_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->dest_port = *(uint16_t *) param;
}

static void
set_listen_port_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->listen_port = *(uint16_t *) param;
}

static void
set_num_clients_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->num_clients = *(int *) param;
}

static void
set_size_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->vector_size = *(int *) param;
}

static void
set_datatype_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;
	const char *str = (const char *) param;

	if (strcmp(str, allreduce_datatype_str[ALLREDUCE_BYTE]) == 0)
		app_config->datatype = ALLREDUCE_BYTE;
	else if (strcmp(str, allreduce_datatype_str[ALLREDUCE_INT]) == 0)
		app_config->datatype = ALLREDUCE_INT;
	else if (strcmp(str, allreduce_datatype_str[ALLREDUCE_FLOAT]) == 0)
		app_config->datatype = ALLREDUCE_FLOAT;
	else if (strcmp(str, allreduce_datatype_str[ALLREDUCE_DOUBLE]) == 0)
		app_config->datatype = ALLREDUCE_DOUBLE;
	else
		APP_EXIT("unknow datatype '%s' was specified", str);
}

static void
set_operation_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;
	const char *str = (const char *) param;

	if (strcmp(str, allreduce_operation_str[ALLREDUCE_SUM]) == 0)
		app_config->operation = ALLREDUCE_SUM;
	else if (strcmp(str, allreduce_operation_str[ALLREDUCE_PROD]) == 0)
		app_config->operation = ALLREDUCE_PROD;
	else
		APP_EXIT("unknow operation '%s' was specified", str);
}

static void
set_batch_size_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->batch_size = *(int *) param;
}

static void
set_num_batches_param(void *config, void *param)
{
	struct allreduce_config *app_config = (struct allreduce_config *) config;

	app_config->num_batches = *(int *) param;
}

void
register_allreduce_params(void)
{
	struct doca_argp_param role_param = {
		.short_flag = "r",
		.long_flag = "role",
		.arguments = NULL,
		.description = "Run DOCA UCX allreduce process as: \"client\" or \"daemon\"",
		.callback = set_role_param,
		.arg_type = DOCA_ARGP_TYPE_STRING,
		.is_mandatory = true,
		.is_cli_only = false
	};
	/* This parameter has affect for client process */
	struct doca_argp_param allreduce_mode_param = {
		.short_flag = "m",
		.long_flag = "mode",
		.arguments = "<allreduce_mode>",
		.description = "Set allreduce mode: \"offloaded\", \"non-offloaded\" (valid for client only)",
		.callback = set_allreduce_mode_param,
		.arg_type = DOCA_ARGP_TYPE_STRING,
		.is_mandatory = false,
		.is_cli_only = false
	};
	struct doca_argp_param dest_port_param = {
		.short_flag = "p",
		.long_flag = "port",
		.arguments = "<port>",
		.description = "Set default destination port of daemons/clients, used for IPs without a port (see '-a' flag)",
		.callback = set_dest_port_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = false,
		.is_cli_only = false
	};
	struct doca_argp_param dest_listen_port_param = {
		.short_flag = "t",
		.long_flag = "listen-port",
		.arguments = "<listen_port>",
		.description = "Set listening port of daemon or client",
		.callback = set_listen_port_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = false,
		.is_cli_only = false
	};
	/* This parameter has affect for daemon proccess */
	struct doca_argp_param num_clients_param = {
		.short_flag = "c",
		.long_flag = "num-clients",
		.arguments = "<num_clients>",
		.description = "Set the number of clients which participate in allreduce operations (valid for daemon only)",
		.callback = set_num_clients_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = false,
		.is_cli_only = false
	};
	struct doca_argp_param size_param = {
		.short_flag = "s",
		.long_flag = "size",
		.arguments = "<size>",
		.description = "Set size of vector to do allreduce for",
		.callback = set_size_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = false,
		.is_cli_only = false
	};
	struct doca_argp_param datatype_param = {
		.short_flag = "d",
		.long_flag = "datatype",
		.arguments = "<datatype>",
		.description = "Set datatype (\"byte\", \"int\", \"float\", \"double\") of vector elements to do allreduce for",
		.callback = set_datatype_param,
		.arg_type = DOCA_ARGP_TYPE_STRING,
		.is_mandatory = false,
		.is_cli_only = false
	};
	struct doca_argp_param operation_param = {
		.short_flag = "o",
		.long_flag = "operation",
		.arguments = "<operation>",
		.description = "Set operation (\"sum\", \"prod\") to do between allreduce vectors",
		.callback = set_operation_param,
		.arg_type = DOCA_ARGP_TYPE_STRING,
		.is_mandatory = false,
		.is_cli_only = false
	};
	/* This parameter has affect for client proccess */
	struct doca_argp_param batch_size_param = {
		.short_flag = "b",
		.long_flag = "batch-size",
		.arguments = "<batch_size>",
		.description = "Set the number of allreduce operations submitted simultaneously (used for handshakes by daemons)",
		.callback = set_batch_size_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = true,
		.is_cli_only = false
	};
	/* This parameter has affect for client proccess */
	struct doca_argp_param num_batches_param = {
		.short_flag = "i",
		.long_flag = "num-batches",
		.arguments = "<num_batches>",
		.description = "Set the number of batches of allreduce operations (used for handshakes by daemons)",
		.callback = set_num_batches_param,
		.arg_type = DOCA_ARGP_TYPE_INT,
		.is_mandatory = true,
		.is_cli_only = false
	};
	struct doca_argp_param dest_ip_str_param = {
		.short_flag = "a",
		.long_flag = "address",
		.arguments = "<ip address>",
		.description = "Set comma-separated list of destination IPv4/IPv6 addresses and ports optionally (<ip_addr>:[<port>]) of daemons or clients",
		.callback = set_dest_ip_str_param,
		.arg_type = DOCA_ARGP_TYPE_STRING,
		.is_mandatory = false,
		.is_cli_only = false
	};

	doca_argp_register_param(&role_param);
	doca_argp_register_param(&allreduce_mode_param);
	doca_argp_register_param(&dest_port_param);
	doca_argp_register_param(&num_clients_param);
	doca_argp_register_param(&size_param);
	doca_argp_register_param(&datatype_param);
	doca_argp_register_param(&operation_param);
	doca_argp_register_param(&batch_size_param);
	doca_argp_register_param(&num_batches_param);
	doca_argp_register_param(&dest_listen_port_param);
	doca_argp_register_param(&dest_ip_str_param);
	doca_argp_register_version_callback(sdk_version_callback);
}

struct allreduce_super_request *
allreduce_super_request_get(const struct allreduce_header *allreduce_header, size_t result_length,
			    void *result_vector)
{
	struct allreduce_super_request *allreduce_super_request;

	/* Check having allreduce super request in the hash */
	allreduce_super_request = g_hash_table_lookup(allreduce_super_requests_hash, &allreduce_header->id);
	if (allreduce_super_request == NULL) {
		if (allreduce_config.role == ALLREDUCE_CLIENT)
			DOCA_LOG_DBG("Starting new operation with id %zu, for vector size: %zu", allreduce_header->id,
				     result_length);
		/* If there is no allreduce super request in the hash, allocate it */
		allreduce_super_request = allreduce_super_request_allocate(allreduce_header, result_length,
										result_vector);
		if (allreduce_super_request == NULL)
			return NULL;

		/* Insert the allocated allreduce super request to the hash */
		g_hash_table_insert(allreduce_super_requests_hash, &allreduce_super_request->header.id,
							allreduce_super_request);
	}

	return allreduce_super_request;
}

static void
connections_cleanup(int num_connections)
{
	int i;

	/* Go over all connections and destroy them by disconnecting */
	for (i = 0; i < num_connections; ++i)
		allreduce_ucx_disconnect(connections[i]);

	free(connections);
}

/* Connects offloaded clients to their daemon and connects daemons/non-offloaded-clients to other daemons/clients */
static int
connections_init(void)
{
	struct allreduce_address *address;
	struct allreduce_ucx_connection *connection;
	int ret, num_connections = 0;

	connections = malloc(allreduce_config.dest_addresses.num * sizeof(*connections));
	if (connections == NULL) {
		DOCA_LOG_ERR("failed to allocate memory to hold array of connections");
		return -1;
	}

	/* Go over peer's addresses and establish connection to the peer using specified address */
	STAILQ_FOREACH(address, &allreduce_config.dest_addresses.list, entry) {
		connection = NULL;
		DOCA_LOG_INFO("Connecting to %s...", address->ip_address_str);
		ret = allreduce_ucx_connect(context, address->ip_address_str, address->port, &connection);
		if (ret < 0) {
			DOCA_LOG_INFO("Failed to establish connection...");
			connections_cleanup(num_connections);
			return -1;
		}
		DOCA_LOG_INFO("Connection established.");

		/* Save connection to the array of connections */
		connections[num_connections++] = connection;
	}

	return num_connections;
}

static guint
g_size_t_hash(gconstpointer v)
{
	return (guint) *(const size_t *)v;
}

static gboolean
g_size_t_equal(gconstpointer v1, gconstpointer v2)
{
	return *((const size_t *)v1) == *((const size_t *)v2);
}

static void
allreduce_super_request_destroy_callback(gpointer data)
{
	allreduce_super_request_destroy(data);
}

void
allreduce_super_request_destroy(struct allreduce_super_request *allreduce_super_request)
{
	if (allreduce_super_request->result_vector_owner)
		free(allreduce_super_request->result_vector);

	free(allreduce_super_request->recv_vectors);
	free(allreduce_super_request);
}

static inline void
allreduce_datatype_memset(void *vector, size_t length)
{
	uint8_t natural_value_byte;
	int natural_value_int;
	float natural_value_float;
	double natural_value_double;
	size_t i;

	if (allreduce_config.operation == ALLREDUCE_SUM) {
		natural_value_byte = 0;
		natural_value_int = 0;
		natural_value_float = 0.;
		natural_value_double = 0.;
	} else {
		assert(allreduce_config.operation == ALLREDUCE_PROD);
		natural_value_byte = 1;
		natural_value_int = 1;
		natural_value_float = 1.;
		natural_value_double = 1.;
	}

	switch (allreduce_config.datatype) {
	case ALLREDUCE_BYTE:
		for (i = 0; i < length; ++i)
			((uint8_t *)vector)[i] = natural_value_byte;
		break;
	case ALLREDUCE_INT:
		for (i = 0; i < length; ++i)
			((int *)vector)[i] = natural_value_int;
		break;
	case ALLREDUCE_FLOAT:
		for (i = 0; i < length; ++i)
			((float *)vector)[i] = natural_value_float;
		break;
	case ALLREDUCE_DOUBLE:
		for (i = 0; i < length; ++i)
			((double *)vector)[i] = natural_value_double;
		break;
	}
}

struct allreduce_super_request *
allreduce_super_request_allocate(const struct allreduce_header *header, size_t length, void *result_vector)
{
	struct allreduce_super_request *allreduce_super_request;

	allreduce_super_request = malloc(sizeof(*allreduce_super_request));
	if (allreduce_super_request == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for allreduce super request");
		return NULL;
	}

	/* Set default values to the fields of the allreduce super requests */
	STAILQ_INIT(&allreduce_super_request->allreduce_requests_list);
	allreduce_super_request->header = *header;
	allreduce_super_request->num_allreduce_requests = 0;
	/*
	 * Count required send & receive vectors between us and peers (daemons or non-offloaded clients).
	 * Also, count +1 operation for completing operations in case of no peers exist.
	 */
	allreduce_super_request->num_allreduce_operations = 2 * allreduce_config.dest_addresses.num + 1;
	allreduce_super_request->result_vector_size = length;
	allreduce_super_request->recv_vector_iter = 0;
	if (result_vector == NULL) {
		/*
		 * result_vector is NULL in case processing Active Message receive operation from peers on daemon or
		 * non-offloaded client side
		 */
		allreduce_super_request->result_vector_owner = 1;
		allreduce_super_request->result_vector = malloc(allreduce_super_request->result_vector_size *
								allreduce_datatype_size[allreduce_config.datatype]);
		if (allreduce_super_request->result_vector == NULL) {
			DOCA_LOG_ERR("failed to allocate memory to hold the allreduce result");
			free(allreduce_super_request);
			return NULL;
		}
		allreduce_datatype_memset(allreduce_super_request->result_vector,
						allreduce_super_request->result_vector_size);
	} else {
		allreduce_super_request->result_vector_owner = 0;
		allreduce_super_request->result_vector = result_vector;
	}

	/* Allocate receive vectors for each connection to be used when doing allreduce between daemons or clients */
	allreduce_super_request->recv_vectors =
			malloc(allreduce_config.dest_addresses.num * sizeof(*allreduce_super_request->recv_vectors));
	if (allreduce_super_request->recv_vectors == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for receive vectors");
		if (allreduce_super_request->result_vector_owner)
			free(allreduce_super_request->result_vector);
		free(allreduce_super_request);
		return NULL;
	}

	return allreduce_super_request;
}

static void
summation(void *dest_vector, void *src_vector, size_t length)
{
	size_t i;

	switch (allreduce_config.datatype) {
	case ALLREDUCE_BYTE:
		for (i = 0; i < length; ++i)
			((uint8_t *)dest_vector)[i] += ((uint8_t *)src_vector)[i];
		break;
	case ALLREDUCE_INT:
		for (i = 0; i < length; ++i)
			((int *)dest_vector)[i] += ((int *)src_vector)[i];
		break;
	case ALLREDUCE_FLOAT:
		for (i = 0; i < length; ++i)
			((float *)dest_vector)[i] += ((float *)src_vector)[i];
		break;
	case ALLREDUCE_DOUBLE:
		for (i = 0; i < length; ++i)
			((double *)dest_vector)[i] += ((double *)src_vector)[i];
		break;
	}
}

static void
product(void *dest_vector, void *src_vector, size_t length)
{
	size_t i;

	switch (allreduce_config.datatype) {
	case ALLREDUCE_BYTE:
		for (i = 0; i < length; ++i)
			((uint8_t *)dest_vector)[i] *= ((uint8_t *)src_vector)[i];
		break;
	case ALLREDUCE_INT:
		for (i = 0; i < length; ++i)
			((int *)dest_vector)[i] *= ((int *)src_vector)[i];
		break;
	case ALLREDUCE_FLOAT:
		for (i = 0; i < length; i++)
			((float *)dest_vector)[i] *= ((float *)src_vector)[i];
		break;
	case ALLREDUCE_DOUBLE:
		for (i = 0; i < length; ++i)
			((double *)dest_vector)[i] *= ((double *)src_vector)[i];
		break;
	}
}

void
allreduce_do_reduce(struct allreduce_super_request *allreduce_super_request, void *allreduce_vector)
{
	switch (allreduce_config.operation) {
	case ALLREDUCE_SUM:
		summation(allreduce_super_request->result_vector, allreduce_vector,
				allreduce_super_request->result_vector_size);
		break;
	case ALLREDUCE_PROD:
		product(allreduce_super_request->result_vector, allreduce_vector,
			allreduce_super_request->result_vector_size);
		break;
	}
}

void
allreduce_request_destroy(struct allreduce_request *allreduce_request)
{
	free(allreduce_request->vector);
	free(allreduce_request);
}

struct allreduce_request *
allreduce_request_allocate(struct allreduce_ucx_connection *connection, const struct allreduce_header *header,
			   size_t length)
{
	struct allreduce_request *allreduce_request;

	allreduce_request = malloc(sizeof(*allreduce_request));
	if (allreduce_request == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for Allreduce request");
		return NULL;
	}

	/* Set default values to the fields of the allreduce request */
	allreduce_request->header = *header;
	allreduce_request->vector = malloc(length * allreduce_datatype_size[allreduce_config.datatype]);
	if (allreduce_request->vector == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for receive vector");
		free(allreduce_request);
		return NULL;
	}

	allreduce_request->vector_size = length / allreduce_datatype_size[allreduce_config.datatype];
	allreduce_request->connection = connection;
	allreduce_request->allreduce_super_request = NULL;

	return allreduce_request;
}

static void
daemon_allreduce_complete_client_operation_callback(void *arg, ucs_status_t status)
{
	struct allreduce_request *allreduce_request = arg;
	struct allreduce_super_request *allreduce_super_request = allreduce_request->allreduce_super_request;

	assert(status == UCS_OK);

	/* Sending completion to the client was completed, release the allreduce request */
	allreduce_request_destroy(allreduce_request);

	--allreduce_super_request->num_allreduce_requests;
	if (allreduce_super_request->num_allreduce_requests == 0) {
		/* All allreduce operations were completed, release allreduce super request */
		allreduce_super_request_destroy(allreduce_super_request);
	}
}

static void
allreduce_complete_operation_callback(void *arg, ucs_status_t status)
{
	struct allreduce_super_request *allreduce_super_request = arg;
	struct allreduce_request *allreduce_request, *tmp_allreduce_request;
	size_t connection_iter;

	if (status != UCS_OK)
		APP_EXIT("Failed to complete an Allreduce scatter/gather. Please check all peers are alive.");

	--allreduce_super_request->num_allreduce_operations;
	/* Check if completed receive and send operations per each connection */
	if (allreduce_super_request->num_allreduce_operations > 0) {
		/* Not all allreduce operations among clients or daemons were completed yet */
		return;
	}

	DOCA_LOG_DBG("Finished 'gathering from peers' stage of request %zu.", allreduce_super_request->header.id);

	/* All allreduce operations among clients or daemons were completed */
	for (connection_iter = 0; connection_iter < allreduce_config.dest_addresses.num; ++connection_iter) {
		/* Do operation among all elements of the received vector */
		allreduce_do_reduce(allreduce_super_request, allreduce_super_request->recv_vectors[connection_iter]);
		free(allreduce_super_request->recv_vectors[connection_iter]);
	}

	/* Remove the allreduce super request from the hash */
	g_hash_table_steal(allreduce_super_requests_hash, &allreduce_super_request->header.id);

	if (allreduce_config.role == ALLREDUCE_CLIENT) {
		assert(STAILQ_EMPTY(&allreduce_super_request->allreduce_requests_list));
		/* Allreduce operation is completed for client, because there is no need to send the result to peers */
		allreduce_super_request_destroy(allreduce_super_request);
		return;
	}

	/* Go over all requests received from the clients and send the result to them */
	STAILQ_FOREACH_SAFE(allreduce_request, &allreduce_super_request->allreduce_requests_list, entry,
						tmp_allreduce_request) {
		/* A completion is sent only by daemons to clients */
		STAILQ_REMOVE(&allreduce_super_request->allreduce_requests_list, allreduce_request,
				allreduce_request, entry);
		allreduce_ucx_am_send(allreduce_request->connection, ALLREDUCE_CTRL_AM_ID,
				      &allreduce_super_request->header, sizeof(allreduce_super_request->header),
				      allreduce_super_request->result_vector,
				      allreduce_super_request->result_vector_size *
					      allreduce_datatype_size[allreduce_config.datatype],
				      daemon_allreduce_complete_client_operation_callback, allreduce_request, NULL);
	}
}

void
allreduce_scatter(struct allreduce_super_request *allreduce_super_request)
{
	size_t i;

	/* Post send operations to exchange allreduce vectors among other daemons/clients */
	for (i = 0; i < allreduce_config.dest_addresses.num; ++i)
		allreduce_ucx_am_send(connections[i], ALLREDUCE_OP_AM_ID, &allreduce_super_request->header,
						sizeof(allreduce_super_request->header),
						allreduce_super_request->result_vector,
						allreduce_super_request->result_vector_size *
						allreduce_datatype_size[allreduce_config.datatype],
						allreduce_complete_operation_callback, allreduce_super_request, NULL);

	DOCA_LOG_DBG("Finished 'scatter' stage for request %zu.", allreduce_super_request->header.id);

	/*
	 * Try to complete the operation, it completes if no other daemons or non-offloaded clients exist or sends were
	 * completed immediately
	 */
	allreduce_complete_operation_callback(allreduce_super_request, UCS_OK);
}

/* AM receive callback which is invoked when the daemon/client receives notification from another daemon/client */
int
allreduce_gather_callback(struct allreduce_ucx_am_desc *am_desc)
{
	struct allreduce_ucx_connection *connection;
	const struct allreduce_header *allreduce_header;
	struct allreduce_super_request *allreduce_super_request;
	void *vector;
	size_t header_length, length, vector_size;

	allreduce_ucx_am_desc_query(am_desc, &connection, (const void **)&allreduce_header, &header_length, &length);

	assert(sizeof(*allreduce_header) == header_length);
	assert(length % allreduce_datatype_size[allreduce_config.datatype] == 0);

	vector_size = length / allreduce_datatype_size[allreduce_config.datatype];

	/* Either find or allocate the allreduce super request to start doing allreduce operations */
	allreduce_super_request = allreduce_super_request_get(allreduce_header, vector_size, NULL);
	if (allreduce_super_request == NULL)
		return -1;

	vector = malloc(allreduce_super_request->result_vector_size *
			allreduce_datatype_size[allreduce_config.datatype]);
	if (vector == NULL) {
		DOCA_LOG_ERR("failed to allocate memory to hold receive vector");
		return -1;
	}

	assert(allreduce_super_request->recv_vector_iter < allreduce_config.dest_addresses.num);

	/* Save vector to the array of receive vectors for futher performing allreduce and releasing it then */
	allreduce_super_request->recv_vectors[allreduce_super_request->recv_vector_iter] = vector;
	++allreduce_super_request->recv_vector_iter;

	/* Continue receiving data to the allocated vector */
	DOCA_LOG_DBG("Received a vector from a peer.");
	allreduce_ucx_am_recv(am_desc, vector, length, allreduce_complete_operation_callback, allreduce_super_request,
			      NULL);
	return 0;
}

static int
allreduce_incoming_handshake_callback(struct allreduce_ucx_am_desc *am_desc)
{
	struct allreduce_ucx_connection *connection;
	const uint8_t *recv_header;
	const uint8_t reply_header = 1;
	char remote_handshake_msg[1024];
	size_t header_length, length;
	struct allreduce_ucx_request *request_p;
	int ret;
	char handshake_msg[1024];
	int handshake_msg_len;

	allreduce_ucx_am_desc_query(am_desc, &connection, (const void **)&recv_header, &header_length, &length);

	/* Get remote handshake message */
	ret = allreduce_ucx_am_recv(am_desc, remote_handshake_msg, length, NULL, NULL, &request_p);
	ret = allreduce_ucx_request_wait(ret, request_p);
	if (ret < 0)
		return ret;
	remote_handshake_msg[length] = '\0';

	/* Create local handshake message */
	handshake_msg_len =
		sprintf(handshake_msg, "-s %zu -d %d -b %zu -i %zu", allreduce_config.vector_size,
			allreduce_config.datatype, allreduce_config.batch_size, allreduce_config.num_batches);
	if (handshake_msg_len < 0)
		APP_EXIT("Failed to generate handshake message (sprintf returned %d)", handshake_msg_len);

	/* Compare settings */
	if (strcmp(handshake_msg, remote_handshake_msg) == 0)
		return 0;

	/* If the sender is a client and we are a daemon, notify the sender of the mismatch */
	if (*recv_header == ALLREDUCE_CLIENT && allreduce_config.role == ALLREDUCE_DAEMON) {
		ret = allreduce_ucx_am_send(connection, ALLREDUCE_HANDSHAKE_AM_ID, &reply_header, 1, handshake_msg,
					    handshake_msg_len, NULL, NULL, &request_p);
		ret = allreduce_ucx_request_wait(ret, request_p);
		if (ret < 0)
			DOCA_LOG_ERR("Failed to replay to handshake message from client.");
	}

	/* Warn user */
	DOCA_LOG_ERR("Configuration mismatch. Us: \"%s\", Other: \"%s\"", handshake_msg, remote_handshake_msg);
	if (allreduce_config.role == ALLREDUCE_CLIENT && allreduce_config.allreduce_mode == ALLREDUCE_OFFLOAD_MODE)
		APP_EXIT("Daemon configuration differ, exiting.");

	DOCA_LOG_CRIT("Please rerun one of the daemons/clients with matching parameters or allreduce will crash.");
	DOCA_LOG_INFO(
		"Daemons and non-offloaded clients cannot guess the correct settings - No terminations is performed.");

	return -1;
}

static int
allreduce_outgoing_handshake(int num_connections)
{
	int i, ret;
	char handshake_msg[1024];
	struct allreduce_ucx_request *request_p;
	uint8_t header = allreduce_config.role;
	int handshake_msg_len =
		sprintf(handshake_msg, "-s %zu -d %d -b %zu -i %zu", allreduce_config.vector_size,
			allreduce_config.datatype, allreduce_config.batch_size, allreduce_config.num_batches);

	if (handshake_msg_len < 0)
		return -1;

	for (i = 0; i < num_connections; ++i) {
		ret = allreduce_ucx_am_send(connections[i], ALLREDUCE_HANDSHAKE_AM_ID, &header, 1, handshake_msg,
					    handshake_msg_len, NULL, NULL, &request_p);
		ret = allreduce_ucx_request_wait(ret, request_p);
		if (ret < 0)
			return ret;
	}

	return 0;
}

static void
communication_destroy(int num_connections)
{
	/* Destroy connections to other clients or daemon in case of client or to other daemons in case of daemon */
	connections_cleanup(num_connections);
	g_hash_table_destroy(allreduce_super_requests_hash);
}

/* The return value indicates how many connections were created */
static int
communication_init(void)
{
	int ret;
	int num_connections;

	/* Allocate hash of allreduce requests to hold submitted operations */
	allreduce_super_requests_hash = g_hash_table_new_full(g_size_t_hash, g_size_t_equal, NULL,
								allreduce_super_request_destroy_callback);
	if (allreduce_super_requests_hash == NULL)
		return -1;

	/* Set handshake message receive handler before starting to accept any connections */
	allreduce_ucx_am_set_recv_handler(context, ALLREDUCE_HANDSHAKE_AM_ID, allreduce_incoming_handshake_callback);

	if ((allreduce_config.role == ALLREDUCE_DAEMON) ||
		(allreduce_config.allreduce_mode == ALLREDUCE_NON_OFFLOADED_MODE)) {
		/*
		 * Setup receive handler for AM messages from daemons or non-offloaded clients which carry allreduce
		 * data to do allreduce for
		 */
		allreduce_ucx_am_set_recv_handler(context, ALLREDUCE_OP_AM_ID, allreduce_gather_callback);

		/* Setup the listener to accept incoming connections from clients/daemons */
		ret = allreduce_ucx_listen(context, allreduce_config.listen_port);
		if (ret < 0) {
			g_hash_table_destroy(allreduce_super_requests_hash);
			return -1;
		}
	}

	/* Initialize connections to other clients or daemon in case of client or to other daemons in case of daemon */
	ret = connections_init();
	if (ret >= 0)
		num_connections = ret;
	else
		return ret;

	/* Send handshake message to peers */
	ret = allreduce_outgoing_handshake(num_connections);
	if (ret < 0) {
		DOCA_LOG_ERR("Failed to perform handshake.");
		communication_destroy(num_connections);
		return -1;
	}

	return num_connections;
}

static void
dest_address_cleanup(void)
{
	struct allreduce_address *address, *tmp_addess;

	/* Go through all addresses saved in the configuration and free the memory allocated to hold them */
	STAILQ_FOREACH_SAFE(address, &allreduce_config.dest_addresses.list, entry, tmp_addess) {
		free(address);
	}
}

static int
dest_addresses_init(void)
{
	int ret = 0;
	char *dest_addresses_str = allreduce_config.dest_addresses.str;
	const char *port_separator;
	char *str;
	size_t ip_addr_length;
	struct allreduce_address *address;

	allreduce_config.dest_addresses.str = NULL;
	allreduce_config.dest_addresses.num = 0;
	STAILQ_INIT(&allreduce_config.dest_addresses.list);

	/* Go over comma-separated list of <IP-address>:[<port>] elements */
	str = strtok(dest_addresses_str, ",");
	while (str != NULL) {
		address = malloc(sizeof(*address));
		if (address == NULL) {
			DOCA_LOG_ERR("failed to allocate memory to hold address");
			ret = -1;
			break;
		}

		/* Parse an element of comma-separated list and insert to the list of peer's addresses */
		port_separator = strchr(str, ':');
		if (port_separator == NULL) {
			/* Port wasn't specified - take port number from -p argument */
			address->port = allreduce_config.dest_port;
			strncpy(address->ip_address_str, str, sizeof(address->ip_address_str) - 1);
			address->ip_address_str[sizeof(address->ip_address_str) - 1] = '\0';
		} else {
			/* Port was specified - take port number from the string of the address */
			address->port = atoi(port_separator + 1);
			ip_addr_length = port_separator - str;
			memcpy(address->ip_address_str, str, ip_addr_length);
			address->ip_address_str[ip_addr_length] = '\0';
		}

		++allreduce_config.dest_addresses.num;
		STAILQ_INSERT_TAIL(&allreduce_config.dest_addresses.list, address, entry);

		str = strtok(NULL, ",");
	}

	if (ret < 0)
		dest_address_cleanup();

	free(dest_addresses_str);
	return ret;
}

int
allreduce_init()
{
	int ret;

	/* Initialize destination addresses specified by a user */
	ret = dest_addresses_init();
	if (ret < 0)
		return ret;

	/* Create context */
	ret = allreduce_ucx_init(&context, ALLREDUCE_MAX_AM_ID);
	if (ret < 0) {
		dest_address_cleanup();
		return ret;
	}

	/* Create comminication-related stuff */
	return communication_init();
}

void
allreduce_destroy(int num_connection)
{
	/* Destroy comminication-related stuff */
	communication_destroy(num_connection);
	/* Destroy UCX context */
	allreduce_ucx_destroy(context);
	/* Destroy destination addresses */
	dest_address_cleanup();
}
