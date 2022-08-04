/*
 * Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <bsd/string.h>
#include <ctype.h>
#include <sys/queue.h>
#include <unistd.h>
#include <linux/types.h>

#include <rte_string_fns.h>

#include <doca_log.h>

#include "telemetry.h"

DOCA_LOG_REGISTER(NETFLOW_TELEMETRY);

struct rte_ring *netflow_pending_ring, *netflow_freelist_ring;

static struct doca_telemetry_netflow_record data_to_send[NETFLOW_QUEUE_SIZE];
static struct doca_telemetry_netflow_record *data_to_send_ptr[NETFLOW_QUEUE_SIZE];

struct doca_telemetry_netflow_flowset_field netflow_template_fields[DOCA_TELEMETRY_NETFLOW_FIELDS_NUM] = {
		{.type = DOCA_TELEMETRY_NETFLOW_IPV4_SRC_ADDR,
			.length = DOCA_TELEMETRY_NETFLOW_IPV4_SRC_ADDR_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IPV4_DST_ADDR,
			.length = DOCA_TELEMETRY_NETFLOW_IPV4_DST_ADDR_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IPV6_SRC_ADDR,
			.length = DOCA_TELEMETRY_NETFLOW_IPV6_SRC_ADDR_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IPV6_DST_ADDR,
			.length = DOCA_TELEMETRY_NETFLOW_IPV6_DST_ADDR_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IPV4_NEXT_HOP,
			.length = DOCA_TELEMETRY_NETFLOW_IPV4_NEXT_HOP_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IPV6_NEXT_HOP,
			.length = DOCA_TELEMETRY_NETFLOW_IPV6_NEXT_HOP_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_INPUT_SNMP,
			.length = DOCA_TELEMETRY_NETFLOW_INPUT_SNMP_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_OUTPUT_SNMP,
			.length = DOCA_TELEMETRY_NETFLOW_OUTPUT_SNMP_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_L4_SRC_PORT,
			.length = DOCA_TELEMETRY_NETFLOW_L4_SRC_PORT_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_L4_DST_PORT,
			.length = DOCA_TELEMETRY_NETFLOW_L4_DST_PORT_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_TCP_FLAGS,
			.length = DOCA_TELEMETRY_NETFLOW_TCP_FLAGS_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_PROTOCOL,
			.length = DOCA_TELEMETRY_NETFLOW_PROTOCOL_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_SRC_TOS,
			.length = DOCA_TELEMETRY_NETFLOW_SRC_TOS_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_SRC_AS,
			.length = DOCA_TELEMETRY_NETFLOW_SRC_AS_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_DST_AS,
			.length = DOCA_TELEMETRY_NETFLOW_DST_AS_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_SRC_MASK,
			.length = DOCA_TELEMETRY_NETFLOW_SRC_MASK_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_DST_MASK,
			.length = DOCA_TELEMETRY_NETFLOW_DST_MASK_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IN_PKTS,
			.length = DOCA_TELEMETRY_NETFLOW_IN_PKTS_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_IN_BYTES,
			.length = DOCA_TELEMETRY_NETFLOW_IN_BYTES_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_FIRST_SWITCHED,
			.length = DOCA_TELEMETRY_NETFLOW_FIRST_SWITCHED_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_LAST_SWITCHED,
			.length = DOCA_TELEMETRY_NETFLOW_LAST_SWITCHED_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_CONNECTION_TRANSACTION_ID,
			.length = DOCA_TELEMETRY_NETFLOW_CONNECTION_TRANSACTION_ID_DEFAULT_LENGTH},
		{.type = DOCA_TELEMETRY_NETFLOW_APPLICATION_NAME,
			.length = DOCA_TELEMETRY_NETFLOW_APPLICATION_NAME_DEFAULT_LENGTH}
};

struct doca_telemetry_netflow_template netflow_template = {
	.field_count = DOCA_TELEMETRY_NETFLOW_FIELDS_NUM,
	.fields = netflow_template_fields
};

static int
get_hostname_ip(char *host_name)
{
	char host[256];
	struct hostent *host_entry = NULL;

	 /* Find the host name */
	if (gethostname(host, sizeof(host)) < 0)  {
		DOCA_LOG_ERR("Gethostname failed");
		return -1;
	}
	/* Find host information */
	host_entry = gethostbyname(host);
	if (host_entry == NULL)
		strlcpy(host_name, host, 64);
	else
		strlcpy(host_name, host_entry->h_name, 64);

	DOCA_LOG_DBG("Host name: %s", host_name);
	return 0;
}

int
send_netflow_record(void)
{
	int ret;
	size_t records_to_send = 0;
	size_t records_sent = 0;
	size_t records_successfully_sent;
	int ring_count = rte_ring_count(netflow_pending_ring);
	static struct doca_telemetry_netflow_record *records[NETFLOW_QUEUE_SIZE];
	/*
	 * Sending the record array
	 * The while loop ensure that all records have been sent, in case just some are sent.
	 * This section should happen periodically with updated the flows.
	 */
	if (ring_count == 0)
		return 0;
	/* We need to dequeue only the records that were enqueued with the allocated memory. */
	records_to_send = rte_ring_dequeue_bulk(netflow_pending_ring, (void **)records, ring_count, NULL);
	while (records_sent < records_to_send) {
		ret = doca_telemetry_netflow_send(&netflow_template, (const void **)(records + records_sent),
				records_to_send - records_sent, &records_successfully_sent);
		if (ret != DOCA_TELEMETRY_OK) {
			DOCA_LOG_ERR("Failed to send Netflow, error=%d", ret);
			return ret;
		}
		records_sent += records_successfully_sent;
	}
	/* Flushing the buffer sends it to the collector */
	flush_telemetry_netflow_source();
	DOCA_LOG_DBG("Successfully sent %lu netflow records with default template.", records_sent);
	if ((int)rte_ring_enqueue_bulk(netflow_freelist_ring, (void **)records, records_sent, NULL) != records_sent) {
		DOCA_LOG_ERR("Placetholder queue mismatch");
		return -1;
	}
	return 0;
}

void
enqueue_netflow_record_to_ring(const struct doca_telemetry_netflow_record *record)
{
	struct doca_telemetry_netflow_record *tmp_record;
	/* To avoid memory corruption when flows are destroyed, we copy the pointers to a
	 *	preallocated pointer inside freelist ring and enqueue it so the main thread
	 *	can send them.
	 */
	if (rte_ring_mc_dequeue(netflow_freelist_ring, (void **)&tmp_record) != 0) {
		DOCA_LOG_DBG("Placeholder queue is empty");
		return;
	}
	*tmp_record = *record;
	if (rte_ring_mp_enqueue(netflow_pending_ring, tmp_record) != 0) {
		DOCA_LOG_DBG("Netflow queue is full");
		return;
	}
}


void
flush_telemetry_netflow_source(void)
{
	doca_telemetry_netflow_flush();
}

void
destroy_netflow_schema_and_source(void)
{
	rte_ring_free(netflow_pending_ring);
	rte_ring_free(netflow_freelist_ring);

	doca_telemetry_netflow_destroy();

}

int
init_netflow_schema_and_source(uint8_t id, char *source_tag)
{
	int res, i;
	char hostname[64];
	struct doca_telemetry_buffer_attr_t buffer = {
		.buffer_size = DOCA_TELEMETRY_DEFAULT_BUFFER_SIZE,
		.data_root   = DOCA_TELEMETRY_DEFAULT_DATA_ROOT,
	};
	struct doca_telemetry_file_write_attr_t file_write = {
		.file_write_enabled = false,
		.max_file_size      = DOCA_TELEMETRY_DEFAULT_FILE_SIZE,
		.max_file_age       = DOCA_TELEMETRY_DEFAULT_FILE_AGE,
	};
	struct doca_telemetry_ipc_attr_t ipc = {
		.ipc_enabled = 1,
		.ipc_sockets_dir = DOCA_TELEMETRY_DEFAULT_IPC_SOCKET_DIR,
	};
	/* Setting the send_attr struct is recommended for debugging only - use DTS otherwise */
	struct doca_telemetry_netflow_send_attr_t netflow = {
		.netflow_collector_addr = "192.168.100.1", /* Bluefield's Rshim interface */
		.netflow_collector_port = NETFLOW_COLLECTOR_PORT,
	};

	struct doca_telemetry_source_name_attr_t source_attr = {
		.source_id = hostname,
		.source_tag = source_tag,
	};

	res = doca_telemetry_netflow_init(id);
	if (res != 0) {
		DOCA_LOG_ERR("Cannot init DOCA netflow");
		goto netflow_exporter_init_failed;
	}

	if (doca_telemetry_netflow_buffer_attr_set(&buffer) != DOCA_TELEMETRY_OK) {
		DOCA_LOG_ERR("Cannot set DOCA Netflow buffer attributes");
		goto netflow_exporter_start_failed;
	}

	if  (doca_telemetry_netflow_file_write_attr_set(&file_write) != DOCA_TELEMETRY_OK) {
		DOCA_LOG_ERR("Cannot set DOCA Netflow write attributes");
		goto netflow_exporter_start_failed;
	}

	if (doca_telemetry_netflow_ipc_attr_set(&ipc) != DOCA_TELEMETRY_OK) {
		DOCA_LOG_ERR("Cannot set DOCA Netflow IPC attributes");
		goto netflow_exporter_start_failed;
	}

	if (doca_telemetry_netflow_send_attr_set(&netflow) != DOCA_TELEMETRY_OK) {
		DOCA_LOG_ERR("Cannot set DOCA Netflow send attributes");
		goto netflow_exporter_start_failed;
	}

	if (get_hostname_ip(hostname) != 0) {
		DOCA_LOG_ERR("Getting hostname failed");
		goto netflow_exporter_start_failed;
	}

	res = doca_telemetry_netflow_start(&source_attr);
	if (res != 0) {
		DOCA_LOG_ERR("Cannot start DOCA netflow");
		goto netflow_exporter_start_failed;
	}
	/* In the Netflow ring scenario, a producer-consumer solution is given where the dpi_worker threads produce
	 * records and enqueues them to a rings struct. The records are consumed by the main thread that dequeues
	 * the records and sends them. This allows avoiding collisions between thread and memory corruption issues.
	 */
	netflow_pending_ring = rte_ring_create("netflow_queue", NETFLOW_QUEUE_SIZE, SOCKET_ID_ANY, RING_F_SC_DEQ);
	netflow_freelist_ring =	rte_ring_create("placeholder_netflow_queue", NETFLOW_QUEUE_SIZE, SOCKET_ID_ANY,
													RING_F_SP_ENQ);
	if (netflow_pending_ring == NULL || netflow_freelist_ring == NULL)
		goto netflow_exporter_start_failed;
	for (i = 0; i < NETFLOW_QUEUE_SIZE; i++)
		data_to_send_ptr[i] = &data_to_send[i];
	if (rte_ring_enqueue_bulk(netflow_freelist_ring, (void **)data_to_send_ptr, NETFLOW_QUEUE_SIZE - 1, NULL) !=
					NETFLOW_QUEUE_SIZE - 1)
		goto netflow_exporter_start_failed;

	return 0;

netflow_exporter_start_failed:
	doca_telemetry_netflow_destroy();
netflow_exporter_init_failed:
	return 1;
}
