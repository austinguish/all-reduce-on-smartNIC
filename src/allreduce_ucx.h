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

#ifndef UCX_CORE_H_
#define UCX_CORE_H_

#include <ucp/api/ucp.h>

struct allreduce_ucx_context;
struct allreduce_ucx_connection;
struct allreduce_ucx_request;
struct allreduce_ucx_am_desc;

typedef void (*allreduce_ucx_callback)(void *arg, ucs_status_t status);
typedef int (*allreduce_ucx_am_callback)(struct allreduce_ucx_am_desc *am_desc);

/***** Requests Processing *****/

int allreduce_ucx_request_wait(int ret, struct allreduce_ucx_request *request);

void allreduce_ucx_request_release(struct allreduce_ucx_request *request);

/***** Active Message send operation *****/

int allreduce_ucx_am_send(struct allreduce_ucx_connection *connection, unsigned int am_id, const void *header,
			  size_t header_length, const void *buffer, size_t length, allreduce_ucx_callback callback,
			  void *arg, struct allreduce_ucx_request **request_p);

/***** Active Message receive operation *****/

int allreduce_ucx_am_recv(struct allreduce_ucx_am_desc *am_desc, void *buffer, size_t length,
			  allreduce_ucx_callback callback, void *arg, struct allreduce_ucx_request **request_p);

void allreduce_ucx_am_desc_query(struct allreduce_ucx_am_desc *am_desc, struct allreduce_ucx_connection **connection,
				 const void **header, size_t *header_length, size_t *length);

void allreduce_ucx_am_set_recv_handler(struct allreduce_ucx_context *context, unsigned int am_id,
				       allreduce_ucx_am_callback callback);

/***** Connection establishment *****/

int allreduce_ucx_connect(struct allreduce_ucx_context *context, const char *dest_ip_str, uint16_t dest_port,
			  struct allreduce_ucx_connection **connection_p);

void allreduce_ucx_disconnect(struct allreduce_ucx_connection *connection);

/***** Main UCX operations *****/

int allreduce_ucx_init(struct allreduce_ucx_context **context_p, unsigned int max_am_id);

void allreduce_ucx_destroy(struct allreduce_ucx_context *context);

int allreduce_ucx_listen(struct allreduce_ucx_context *context, uint16_t port);

void allreduce_ucx_progress(struct allreduce_ucx_context *context);

#endif /** UCX_CORE_H_ */
