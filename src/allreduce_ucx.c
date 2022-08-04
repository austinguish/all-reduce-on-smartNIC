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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <glib.h>

#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/queue.h>
#include <string.h>
#include <assert.h>

#include <doca_log.h>

#include "allreduce_ucx.h"

DOCA_LOG_REGISTER(ALLREDUCE::UCX);

struct allreduce_ucx_am_callback_info {
	struct allreduce_ucx_context *context;	/* Pointer to UCX context */
	allreduce_ucx_am_callback callback;	/* Callback which should be invoked upon receiving AM */
};

struct allreduce_ucx_context {
	ucp_context_h context;			/* Holds a UCP communication instance's global information */
	ucp_worker_h worker;			/* Holds local communication resource and the progress engine
						 * associated with it
						 */
	ucp_listener_h listener;		/* Handle for listening on a specific address and accepting
						 * incoming connections
						 */
	GHashTable *ep_to_connections_hash;	/* Hash Table to map active EP to its active connection */
	unsigned int max_am_id;			/* Maximum Active Message (AM) identifier utilized by the user */
	struct allreduce_ucx_am_callback_info *am_callback_infos;	/* AM callback which was specified by a user */
};

struct allreduce_ucx_connection {
	struct allreduce_ucx_context *context;          /* Pointer to the context which owns this connection */
	ucp_ep_h ep;                                    /* Endpoint that is connected to a remote worker */
	struct sockaddr_storage *destination_address;   /* Address of the peer */
};

enum allreduce_ucx_op {
	ALLREDUCE_UCX_UNKNOWN_OP,	/* Unknown UCX operation */
	ALLREDUCE_UCX_AM_SEND,		/* Active Message (AM) send operation */
	ALLREDUCE_UCX_AM_RECV_DATA	/* Active Message (AM) receive data operation */
};

struct allreduce_ucx_request {
	allreduce_ucx_callback callback;		/* Completion callback which was specified by a user */
	void *arg;					/* Argument which should be passed to the compeltion callback
							 */
	struct allreduce_ucx_connection *connection;	/* Owner of UCX request */
	int log_err;					/* Indicates whether error message should be printed in case
							 * error detected
							 */
	ucs_status_t status;				/* Current status of the operation */
	enum allreduce_ucx_op op;			/* Operation type */
};

struct allreduce_ucx_am_desc {
	struct allreduce_ucx_connection *connection;	/* Pointer to the connection on which this AM operation
							 * was received
							 */
	const void *header;				/* Header got from AM callback */
	size_t header_length;				/* Length of the header got from AM callback */
	void *data_desc;				/* Pointer to the descriptor got from AM callback. In case of
							 * Rendezvous, it is not the actual data, but only a data
							 * descriptor
							 */
	size_t length;					/* Length of the received data */
	uint64_t flags;					/* AM operation flags */
};

static const char * const allreduce_ucx_op_str[] = {
	[ALLREDUCE_UCX_AM_SEND] = "ucp_am_send_nbx",		/* Name of Active Message (AM) send operation */
	[ALLREDUCE_UCX_AM_RECV_DATA] = "ucp_am_recv_data_nbx"	/* Name of Active Message (AM) receive data operation
								 */
};

static GHashTable *active_connections_hash;

/***** Requests Processing *****/

static void
request_init(void *request)
{
	struct allreduce_ucx_request *r = (struct allreduce_ucx_request *)request;

	/* Initialize all fields of UCX request by default values */
	r->connection = NULL;
	r->callback = NULL;
	r->arg = NULL;
	r->op = ALLREDUCE_UCX_UNKNOWN_OP;
	r->status = UCS_INPROGRESS;
	r->log_err = 1;
}

static void
request_release(void *request)
{
	/* Reset UCP request to the initial state */
	request_init(request);
	/* Free UCP request */
	ucp_request_free(request);
}

static inline void
user_request_complete(allreduce_ucx_callback callback, void *arg, struct allreduce_ucx_request **request_p,
		      ucs_status_t status)
{
	if (callback != NULL) {
		/* Callback was specified by a user, invoke it */
		callback(arg, status);
	}

	if (request_p != NULL) {
		/* Storage for request was specified by a user, set it to NULL, because operation was already completed */
		*request_p = NULL;
	}
}

int
request_process(struct allreduce_ucx_connection *connection, enum allreduce_ucx_op op, ucs_status_ptr_t ptr_status,
		allreduce_ucx_callback callback, void *arg, int log_err, struct allreduce_ucx_request **request_p)
{
	const char *what = allreduce_ucx_op_str[op];
	struct allreduce_ucx_request *r = NULL;
	ucs_status_t status;

	if (ptr_status == NULL) {
		/* Operation was completed successfully */
		user_request_complete(callback, arg, request_p, UCS_OK);
		if (request_p != NULL)
			*request_p = NULL;
		return 0;
	} else if (UCS_PTR_IS_ERR(ptr_status)) {
		/* Operation was completed with the error */
		status = UCS_PTR_STATUS(ptr_status);
		if (log_err) {
			/* Requested to print an error */
			DOCA_LOG_ERR("%s failed with status: %s", what, ucs_status_string(status));
		}
		/* Complete operation and provide the error status */
		user_request_complete(callback, arg, request_p, status);
		return -1;
	}

	/* Got pointer to request */
	r = (struct allreduce_ucx_request *)ptr_status;
	if (r->status != UCS_INPROGRESS) {
		/* Already completed by "common_request_callback" */

		assert(r->op == op);

		/* Complete operation and provide the status */
		status = r->status;
		user_request_complete(callback, arg, request_p, status);
		/* Release the request */
		request_release(r);

		if (status != UCS_OK) {
			DOCA_LOG_ERR("%s failed with status: %s", what, ucs_status_string(status));
			return -1;
		}
	} else {
		/* Will be completed by "common_request_callback", initialize the request */

		assert(r->op == ALLREDUCE_UCX_UNKNOWN_OP);

		r->callback = callback;
		r->connection = connection;
		r->arg = arg;
		r->op = op;
		r->log_err = log_err;

		if (request_p != NULL) {
			/* If it was requested by a user, provide the request to wait on */
			*request_p = r;
		}
	}

	return 0;
}

static inline void
common_request_callback(void *request, ucs_status_t status, void *user_data, enum allreduce_ucx_op op)
{
	struct allreduce_ucx_request *r = (struct allreduce_ucx_request *)request;

	/* Save completion status */
	r->status = status;

	if (r->connection != NULL) {
		/* Already processed by "request_process" */
		if (r->callback != NULL) {
			/* Callback was specified by a user, invoke it */
			r->callback(r->arg, status);
			/* Release the request */
			request_release(request);
		} else {
			/* User is responsible to check if the request completed or not and release the request then */
		}
	} else {
		assert(r->op == ALLREDUCE_UCX_UNKNOWN_OP);

		/* Not processed by "request_process" */
		r->connection = user_data;
		r->op = op;
	}
}

int
allreduce_ucx_request_wait(int ret, struct allreduce_ucx_request *request)
{
	if (ret < 0 && request != NULL) {
		/* Operation was completed with error */
		DOCA_LOG_ERR("%p failed: %s", allreduce_ucx_op_str[request->op], ucs_status_string(request->status));
	} else if (request != NULL) {
		while (request->status == UCS_INPROGRESS) {
			/* Progress UCX context until compeltion status is in-progress */
			allreduce_ucx_progress(request->connection->context);
		}

		if (request->status != UCS_OK) {
			/* Operation failed */
			if (request->log_err) {
				/* Print error if requested by a caller */
				DOCA_LOG_ERR("%s failed: %s", allreduce_ucx_op_str[request->op],
						ucs_status_string(request->status));
			}
			ret = -1;
		}

		/* Release the request */
		allreduce_ucx_request_release(request);
	}

	return ret;
}

void
allreduce_ucx_request_release(struct allreduce_ucx_request *request)
{
	request_release(request);
}

/***** Active Message send operation *****/

/* Active Message (AM) send callback */
static void
am_send_request_callback(void *request, ucs_status_t status, void *user_data)
{
	common_request_callback(request, status, user_data, ALLREDUCE_UCX_AM_SEND);
}

static int
am_send(struct allreduce_ucx_connection *connection, unsigned int am_id, int log_err, const void *header,
	size_t header_length, const void *buffer, size_t length, allreduce_ucx_callback callback, void *arg,
	struct allreduce_ucx_request **request_p)
{
	ucp_request_param_t param = {
		/* Completion callback, user data and flags are specified */
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_FLAGS,
		/* Send completion callback */
		.cb.send = am_send_request_callback,
		/* User data is the pointer of the connection on which the operation will posted */
		.user_data = connection,
		/* Force passing UCP EP of the connection to the AM receive handler on a receiver side */
		.flags = UCP_AM_SEND_FLAG_REPLY
	};
	ucs_status_ptr_t status_ptr;

	/* Submit AM send operation */
	status_ptr = ucp_am_send_nbx(connection->ep, am_id, header, header_length, buffer, length, &param);
	/* Process 'status_ptr' */
	return request_process(connection, ALLREDUCE_UCX_AM_SEND, status_ptr, callback, arg, log_err, request_p);
}

int
allreduce_ucx_am_send(struct allreduce_ucx_connection *connection, unsigned int am_id, const void *header,
			size_t header_length, const void *buffer, size_t length, allreduce_ucx_callback callback,
			void *arg, struct allreduce_ucx_request **request_p)
{
	if (g_hash_table_lookup(active_connections_hash, connection) == NULL) {
		DOCA_LOG_WARN("Send to a disconnected endpoint was requested.");
		if (request_p != NULL)
			*request_p = NULL;
		return -1; /* already been disconnected */
	}
	return am_send(connection, am_id, 1, header, header_length, buffer, length, callback, arg, request_p);
}

/***** Active Message receive operation *****/

/* Active Message (AM) receive callback */
static void
am_recv_data_request_callback(void *request, ucs_status_t status, size_t length, void *user_data)
{
	common_request_callback(request, status, user_data, ALLREDUCE_UCX_AM_RECV_DATA);
}

int
allreduce_ucx_am_recv(struct allreduce_ucx_am_desc *am_desc, void *buffer, size_t length,
			allreduce_ucx_callback callback, void *arg, struct allreduce_ucx_request **request_p)
{
	struct allreduce_ucx_connection *connection = am_desc->connection;
	struct allreduce_ucx_context *context = connection->context;
	ucp_request_param_t param = {
		/* Completion callback and user data are specified */
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA,
		/* Completion callback */
		.cb.recv_am = am_recv_data_request_callback,
		/* User data is context which owns the receive operation */
		.user_data = context
	};
	ucs_status_ptr_t status_ptr;

	if (am_desc->flags & UCP_AM_RECV_ATTR_FLAG_RNDV) {
		/* if the received AM descriptor is just a notification about Rendezvous, start receiving the whole data */
		status_ptr = ucp_am_recv_data_nbx(context->worker, am_desc->data_desc, buffer, length, &param);
	} else {
		/* The whole data was read, just copy it the user's buffer */
		status_ptr = NULL;
		memcpy(buffer, am_desc->data_desc, MIN(length, am_desc->length));
	}

	/* Process 'status_ptr' */
	return request_process(connection, ALLREDUCE_UCX_AM_RECV_DATA, status_ptr, callback, arg, 1, request_p);
}

void
allreduce_ucx_am_desc_query(struct allreduce_ucx_am_desc *am_desc, struct allreduce_ucx_connection **connection,
			    const void **header, size_t *header_length, size_t *length)
{
	*connection = am_desc->connection;
	*header = am_desc->header;
	*header_length = am_desc->header_length;
	*length = am_desc->length;
}

/* Proxy to handle AM receive operation and call user's callback */
static ucs_status_t
am_recv_callback(void *arg, const void *header, size_t header_length, void *data_desc, size_t length,
			const ucp_am_recv_param_t *param)
{
	struct allreduce_ucx_am_callback_info *callback_info = arg;
	struct allreduce_ucx_context *context = callback_info->context;
	ucp_ep_h ep = param->reply_ep;
	struct allreduce_ucx_connection *connection;
	struct allreduce_ucx_am_desc am_desc;

	/* Try to find connection in the hash of the connections where key is the UCP EP */
	connection = g_hash_table_lookup(context->ep_to_connections_hash, ep);
	assert(connection != NULL);

	/* Fill AM descriptor which will be passed to the user and used to fetch the data then by
	 * 'allreduce_ucx_am_recv'
	 */
	am_desc.connection = connection;
	am_desc.flags = param->recv_attr;
	am_desc.header = header;
	am_desc.header_length = header_length;
	am_desc.data_desc = data_desc;
	am_desc.length = length;

	/* Invoke user's callback specified for the AM ID */
	callback_info->callback(&am_desc);
	return UCS_OK;
}

static void
am_set_recv_handler_common(ucp_worker_h worker, unsigned int am_id, ucp_am_recv_callback_t cb, void *arg)
{
	ucp_am_handler_param_t param = {
		/* AM identifier, callback and argument are set */
		.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB |
				UCP_AM_HANDLER_PARAM_FIELD_ARG,
		/* Active Message (AM) identifier */
		.id = am_id,
		/* User's callback which should be called upon receiving */
		.cb = cb,
		/* User's argument which should be passed to user's callback upon receiving */
		.arg = arg
	};

	/* Specify AM receive handler to the UCP worker */
	ucp_worker_set_am_recv_handler(worker, &param);
}

void
allreduce_ucx_am_set_recv_handler(struct allreduce_ucx_context *context, unsigned int am_id,
					allreduce_ucx_am_callback callback)
{
	if (context->am_callback_infos == NULL) {
		/* Array of AM callback infos wasn't allocated yet, allocate it now */
		context->am_callback_infos = malloc((context->max_am_id + 1) * sizeof(*context->am_callback_infos));
		if (context->am_callback_infos == NULL) {
			DOCA_LOG_ERR("failed to allocate memory to hold AM callbacks");
			return;
		}
	}

	/* Save user's callback for futher invoking it then upon receiving data */
	context->am_callback_infos[am_id].context = context;
	context->am_callback_infos[am_id].callback = callback;
	am_set_recv_handler_common(context->worker, am_id, am_recv_callback, &context->am_callback_infos[am_id]);
}

/***** Connection establishment *****/

/* Active Message (AM) callback to receive connection check message */
static ucs_status_t
am_connection_check_recv_callback(void *arg, const void *header, size_t header_length, void *data, size_t length,
					const ucp_am_recv_param_t *param)
{
	assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
	return UCS_OK;
}

static void
disconnect_common(struct allreduce_ucx_connection *connection, uint32_t flags)
{
	struct allreduce_ucx_context *context = connection->context;
	ucp_request_param_t close_params = {
		/* Indicate that flags parameter is specified */
		.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
		/* UCP EP closure flags */
		.flags = flags
	};
	ucs_status_t status;
	ucs_status_ptr_t close_req;

	if (connection->ep == NULL) {
		/* Disconnection has already been scheduled */
		return;
	}

	g_hash_table_remove(active_connections_hash, connection);
	g_hash_table_steal(context->ep_to_connections_hash, connection->ep);

	/* Close request is equivalent to an async-handler to know the close operation status */
	close_req = ucp_ep_close_nbx(connection->ep, &close_params);
	if (UCS_PTR_IS_PTR(close_req)) {
		/* Wait completion of UCP EP close operation */
		do {
			/* Progress UCP worker */
			ucp_worker_progress(context->worker);
			status = ucp_request_check_status(close_req);
		} while (status == UCS_INPROGRESS);
		/* Free UCP request */
		ucp_request_free(close_req);
	}

	/* Set UCP EP to NULL to catch possible use after UCP EP closure */
	connection->ep = NULL;
}

static inline void
connection_deallocate(struct allreduce_ucx_connection *connection)
{
	free(connection->destination_address);
	free(connection);
}

static void
destroy_connection_callback(gpointer data)
{
	struct allreduce_ucx_connection *connection = data;

	disconnect_common(connection, UCP_EP_CLOSE_FLAG_FORCE);
	connection_deallocate(connection);
}

static struct allreduce_ucx_connection *
connection_allocate(struct allreduce_ucx_context *context, ucp_ep_params_t *ep_params)
{
	struct allreduce_ucx_connection *connection;

	connection = malloc(sizeof(*connection));
	if (connection == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for connection");
		return NULL;
	}

	connection->ep = NULL;
	connection->context = context;

	if (ep_params->flags & UCP_EP_PARAMS_FLAGS_CLIENT_SERVER) {
		/* Allocate memory to hold destination address which could be used for reconnecting */
		connection->destination_address = malloc(sizeof(*connection->destination_address));
		if (connection->destination_address == NULL) {
			DOCA_LOG_ERR("failed to allocate memory to hold destination address");
			free(connection);
			return NULL;
		}

		/* Fill destination address by socket address used for conenction establishment */
		*connection->destination_address = *(struct sockaddr_storage *)ep_params->sockaddr.addr;
	} else {
		connection->destination_address = NULL;
	}

	return connection;
}

/* Forward declaration */
static void
error_callback(void *arg, ucp_ep_h ep, ucs_status_t status);

static int
connect_common(struct allreduce_ucx_context *context, ucp_ep_params_t *ep_params,
		struct allreduce_ucx_connection **connection_p)
{
	struct allreduce_ucx_connection *connection;
	ucs_status_t status;

	if (*connection_p == NULL) {
		/* It is normal connection establishment - allocate the new connection object */
		connection = connection_allocate(context, ep_params);
		if (connection == NULL)
			return -1;
	} else {
		/* User is reconnecting - use connection from a passed pointer and do reconnection */
		connection = *connection_p;
		assert(connection->context == context);
	}

	/* Error handler and error handling mode are specified */
	ep_params->field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLER | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
	/* Error handling PEER mode is needed to detect disconnection of clients on daemon and close the endpoint */
	ep_params->err_mode = UCP_ERR_HANDLING_MODE_PEER;
	/* Error callback */
	ep_params->err_handler.cb = error_callback;
	/* Argument which will be passed to error callback */
	ep_params->err_handler.arg = connection;

	assert(connection->ep == NULL);

	/* Create UCP EP */
	status = ucp_ep_create(context->worker, ep_params, &connection->ep);
	if (status != UCS_OK) {
		DOCA_LOG_ERR("failed to create UCP endpoint: %s", ucs_status_string(status));
		if (*connection_p == NULL) {
			/* Destroy only if allocated here */
			connection_deallocate(connection);
		}
		return -1;
	}

	/* Insert the new connection to the context's hash of connections */
	g_hash_table_insert(context->ep_to_connections_hash, connection->ep, connection);
	g_hash_table_insert(active_connections_hash, connection, connection);

	*connection_p = connection;
	return 0;
}

static int
sockaddr_connect(struct allreduce_ucx_context *context, const struct sockaddr_storage *dst_saddr,
		 struct allreduce_ucx_connection **connection_p)
{
	ucp_ep_params_t ep_params = {
		/* Flags and socket address are specified */
		.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR,
		/* Client-server connection establishment mode */
		.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
		/* Peer's socket address */
		.sockaddr.addr = (const struct sockaddr *)dst_saddr,
		/* Size of socket address */
		.sockaddr.addrlen = sizeof(*dst_saddr)
	};

	/* Connect to a peer */
	return connect_common(context, &ep_params, connection_p);
}

/* Callback which is invoked upon error detection */
static void
error_callback(void *arg, ucp_ep_h ep, ucs_status_t status)
{
	struct allreduce_ucx_connection *connection = (struct allreduce_ucx_connection *)arg;
	int res;

	/* Disconnect from a peer forcibly */
	disconnect_common(connection, UCP_EP_CLOSE_FLAG_FORCE);
	if (connection->destination_address == NULL) {
		/* If the connection was created from callback - can free the memory */
		free(connection);
		return;
	}

	/* Reconnect to the peer */
	res = sockaddr_connect(connection->context, connection->destination_address, &connection);
	if (res < 0) {
		/* Can't reconnect - print the error message */
		DOCA_LOG_ERR("Connection to peer/daemon broke and attempts to reconnect fail.");
		connection_deallocate(connection);
	}
}

static int
set_sockaddr(const char *ip_str, uint16_t port, struct sockaddr_storage *saddr)
{
	struct sockaddr_in *sa_in = (struct sockaddr_in *)saddr;
	struct sockaddr_in6 *sa_in6 = (struct sockaddr_in6 *)saddr;

	/* Try to convert string representation of the IPv4 address to the socket address */
	if (inet_pton(AF_INET, ip_str, &sa_in->sin_addr) == 1) {
		/* Success - set family and port */
		sa_in->sin_family = AF_INET;
		sa_in->sin_port = htons(port);
		return 0;
	}

	/* Try to convert string representation of the IPv6 address to the socket address */
	if (inet_pton(AF_INET6, ip_str, &sa_in6->sin6_addr) == 1) {
		/* Success - set family and port */
		sa_in6->sin6_family = AF_INET6;
		sa_in6->sin6_port = htons(port);
		return 0;
	}

	DOCA_LOG_ERR("invalid address: '%s'", ip_str);
	return -1;
}

static const char *
sockaddr_str(const struct sockaddr *saddr, size_t addrlen, char *buf, size_t buf_len)
{
	uint16_t port;

	if (saddr->sa_family != AF_INET)
		return "<unknown address family>";

	switch (saddr->sa_family) {
	case AF_INET:
		/* IPv4 address */
		inet_ntop(AF_INET, &((const struct sockaddr_in *)saddr)->sin_addr, buf, buf_len);
		port = ntohs(((const struct sockaddr_in *)saddr)->sin_port);
		break;
	case AF_INET6:
		/* IPv6 address */
		inet_ntop(AF_INET6, &((const struct sockaddr_in6 *)saddr)->sin6_addr, buf, buf_len);
		port = ntohs(((const struct sockaddr_in6 *)saddr)->sin6_port);
		break;
	default:
		return "<invalid address>";
	}

	snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ":%u", port);
	return buf;
}

int
allreduce_ucx_connect(struct allreduce_ucx_context *context, const char *dest_ip_str, uint16_t dest_port,
		      struct allreduce_ucx_connection **connection_p)
{
	struct sockaddr_storage dst_saddr;
	int ret;
	uint8_t dummy[0];
	struct allreduce_ucx_request *request;

	assert(dest_ip_str != NULL);
	/* Set IP address and port specified by a user to the socket address */
	ret = set_sockaddr(dest_ip_str, dest_port, &dst_saddr);
	if (ret < 0)
		return ret;

	*connection_p = NULL;

	/* Connect to the peer using sockeet address generated above */
	ret = sockaddr_connect(context, &dst_saddr, connection_p);
	if (ret < 0)
		return ret;

	/* Try sending connection check AM to make sure the new UCP EP is successfully connected to the peer */
	do {
		/* If sending AM fails, reconnection will be done form the error callback */
		request = NULL;
		ret = am_send(*connection_p, context->max_am_id + 1, 0, &dummy, 0, NULL, 0, NULL, NULL, &request);
		if (ret == 0)
			ret = allreduce_ucx_request_wait(ret, request);
	} while (ret < 0);

	return ret;
}

/* Callback which is invoked upon receiving incoming connection */
static void
connect_callback(ucp_conn_request_h conn_req, void *arg)
{
	struct allreduce_ucx_context *context = (struct allreduce_ucx_context *)arg;
	ucp_ep_params_t ep_params = {
		/* Connection request is specified */
		.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
		/* Connection request */
		.conn_request = conn_req
	};
	ucp_conn_request_attr_t conn_req_attr = {
		/* Request getting the client's address which is an initiator of the connection */
		.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR
	};
	struct allreduce_ucx_connection *connection = NULL;
	char buf[128];
	ucs_status_t status;

	/* Query connection request information */
	status = ucp_conn_request_query(conn_req, &conn_req_attr);
	if (status == UCS_OK) {
		DOCA_LOG_DBG("Got new connection request %p from %s", conn_req,
				sockaddr_str((const struct sockaddr *) &conn_req_attr.client_address,
						sizeof(conn_req_attr.client_address), buf, sizeof(buf)));
	} else {
		DOCA_LOG_ERR("Got new connection request %p, connection request query failed: %s", conn_req,
						ucs_status_string(status));
	}

	/* Connect to the peer by accepting the incoming connection */
	connect_common(context, &ep_params, &connection);
}

void
allreduce_ucx_disconnect(struct allreduce_ucx_connection *connection)
{
	if (g_hash_table_lookup(active_connections_hash, connection) == NULL)
		return;  /* already been disconnected */

	/* Normal disconnection from a peer with flushing all operations */
	disconnect_common(connection, 0);
	connection_deallocate(connection);
}

/***** Main UCX operations *****/

int
allreduce_ucx_init(struct allreduce_ucx_context **context_p, unsigned int max_am_id)
{
	ucp_params_t context_params = {
		/* Features, request initialize callback and request size are specified */
		.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_SIZE,
		/* Request support for Active messages (AM) in a UCP context */
		.features = UCP_FEATURE_AM,
		/* Function which will be invoked to fill UCP request upon allocation */
		.request_init = request_init,
		/* Size of UCP request */
		.request_size = sizeof(struct allreduce_ucx_request)
	};
	ucp_worker_params_t worker_params = {
		/* Thread mode is specified */
		.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
		/* UCP worker progress and all send/receive operations must be called from a single thread at the same
		 * time
		 */
		.thread_mode = UCS_THREAD_MODE_SINGLE
	};
	ucs_status_t status;
	struct allreduce_ucx_context *context;

	context = malloc(sizeof(*context));
	if (context == NULL) {
		DOCA_LOG_ERR("failed to allocate memory for UCX context");
		return -1;
	}

	context->am_callback_infos = NULL;
	context->listener = NULL;

	/* Save maximum AM ID which will be specified by the user */
	context->max_am_id = max_am_id;

	/* Allocate hash to hold all connections created by user or accepted from a peer */
	context->ep_to_connections_hash =
		g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, destroy_connection_callback);
	if (context->ep_to_connections_hash == NULL) {
		free(context);
		return -1;
	}
	active_connections_hash = g_hash_table_new(g_direct_hash, g_direct_equal);
	if (active_connections_hash == NULL) {
		g_hash_table_destroy(context->ep_to_connections_hash);
		free(context);
		return -1;
	}

	/* UCP has default config that is set by env vars, we don't need to change it, so using NULL */
	status = ucp_init(&context_params, NULL, &context->context);
	if (status != UCS_OK) {
		DOCA_LOG_ERR("failed to create UCP context: %s", ucs_status_string(status));
		g_hash_table_destroy(active_connections_hash);
		g_hash_table_destroy(context->ep_to_connections_hash);
		free(context);
		return -1;
	}

	/* Create UCP worker */
	status = ucp_worker_create(context->context, &worker_params, &context->worker);
	if (status != UCS_OK) {
		DOCA_LOG_ERR("failed to create UCP worker: %s", ucs_status_string(status));
		ucp_cleanup(context->context);
		g_hash_table_destroy(active_connections_hash);
		g_hash_table_destroy(context->ep_to_connections_hash);
		free(context);
		return -1;
	}

	/* Use 'max_am_id + 1' to set AM callback to receive connection check message */
	am_set_recv_handler_common(context->worker, context->max_am_id + 1, am_connection_check_recv_callback, NULL);

	*context_p = context;

	return 0;
}

void
allreduce_ucx_destroy(struct allreduce_ucx_context *context)
{
	/* Destroy all created connections inside hash destroy operation */
	g_hash_table_destroy(context->ep_to_connections_hash);
	/* Destroy this table after the above because cleanup method for values in the above table uses both tables. */
	g_hash_table_destroy(active_connections_hash);

	if (context->listener != NULL) {
		/* Destroy UCP listener if it was created by a user */
		ucp_listener_destroy(context->listener);
	}

	/* Destroy UCP worker */
	ucp_worker_destroy(context->worker);
	/* Destroy UCP context */
	ucp_cleanup(context->context);

	free(context->am_callback_infos);
	free(context);
}

int
allreduce_ucx_listen(struct allreduce_ucx_context *context, uint16_t port)
{
	/* Listen on any IPv4 address and the user-specified port */
	const struct sockaddr_in listen_addr = {
		/* Set IPv4 address family */
		.sin_family = AF_INET,
		.sin_addr = {
			/* Set any address */
			.s_addr = INADDR_ANY
		},
		/* Set port from the user */
		.sin_port = htons(port)
	};
	ucp_listener_params_t listener_params = {
		/* Socket address and conenction handler are specified */
		.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
		/* Listen address */
		.sockaddr.addr = (const struct sockaddr *)&listen_addr,
		/* Size of listen address */
		.sockaddr.addrlen = sizeof(listen_addr),
		/* Incoming connection handler */
		.conn_handler.cb = connect_callback,
		/* UCX context which is owner of the connection */
		.conn_handler.arg = context
	};
	ucs_status_t status;

	/* Create UCP listener to accept incoming connections */
	status = ucp_listener_create(context->worker, &listener_params, &context->listener);
	if (status != UCS_OK) {
		DOCA_LOG_ERR("Failed to create UCP listener: %s", ucs_status_string(status));
		return -1;
	}

	return 0;
}

void
allreduce_ucx_progress(struct allreduce_ucx_context *context)
{
	/* Progress send and receive operations on UCP worker */
	ucp_worker_progress(context->worker);
}
