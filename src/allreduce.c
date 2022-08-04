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

#include <assert.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <float.h>

#include "allreduce_ucx.h"
#include "allreduce_core.h"
#include "allreduce_daemon.h"
#include "allreduce_client.h"

DOCA_LOG_REGISTER(ALLREDUCE);

int
main(int argc, char **argv)
{
	struct doca_argp_program_general_config *doca_general_config;
	struct doca_argp_program_type_config type_config = {
		.is_dpdk = false,
		.is_grpc = false
	};
	int ret, num_connections;

	/* Parse cmdline/json arguments */
	doca_argp_init("allreduce", &type_config, &allreduce_config);
	register_allreduce_params();
	doca_argp_start(argc, argv, &doca_general_config);

	ret = allreduce_init();
	if (ret < 0) {
		DOCA_LOG_ERR("Failded to init UCX or failed to connect to a given address.");
		doca_argp_destroy();
		return ret;
	}
	DOCA_LOG_INFO("Successfully connected to all given address.");

	num_connections = ret;
	if (allreduce_config.role == ALLREDUCE_CLIENT) {
		if (num_connections == 0) {
			/* Nothing to do */
			return 0;
		} else if ((num_connections > 1) && (allreduce_config.allreduce_mode == ALLREDUCE_OFFLOAD_MODE)) {
			DOCA_LOG_ERR("number of client's peers in offloaded mode should be 1 instead of %d",
					num_connections);
			allreduce_destroy(num_connections);
			doca_argp_destroy();
			return -1;
		}
	}

	/* Run required code depending on the type of the process */
	if (allreduce_config.role == ALLREDUCE_DAEMON)
		daemon_run();
	else
		client_run();

	/* Destroy argument aprser */
	doca_argp_destroy();
	allreduce_destroy(num_connections);
	return ret;
}
