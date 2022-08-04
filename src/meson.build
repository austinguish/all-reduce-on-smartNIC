#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

if get_option('enable_ucx_support') == false
	warning('Skipping compilation of DOCA Application - @0@.'.format(APP_NAME))
	subdir_done()
endif

app_dependencies += dependency('doca-argp')
app_dependencies += dependency('doca-common')
app_dependencies += dependency('ucx')
app_dependencies += dependency('glib-2.0')

app_srcs += [
	APP_NAME + '.c',
	APP_NAME + '_core.c',
	APP_NAME + '_daemon.c',
	APP_NAME + '_client.c',
	APP_NAME + '_ucx.c',
	common_dir_path + '/utils.c'
]

executable(DOCA_PREFIX + APP_NAME, app_srcs,
	dependencies : app_dependencies,
	include_directories : app_inc_dirs,
	install: false)
