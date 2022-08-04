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

project('DOCA_APPLICATIONS', 'C', 'CPP',
	# Get version number from file.
	version: run_command(find_program('cat'), files('VERSION'), check: true).stdout().strip(),
	license: 'BSD',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.56.0'
)

DOCA_PREFIX = 'doca_'

# With enabling GPU support, languages variable will contain CUDA language
languages = ['c', 'cpp']
if get_option('enable_gpu_support')
	# Enables ifdef for GPU code
	add_project_arguments('-D GPU_SUPPORT', language: languages)
	add_languages('CUDA')
	languages += ['cuda']
	cuda = import('unstable-cuda')
	nvcc = meson.get_compiler('cuda')
	nvcc_flags = cuda.nvcc_arch_flags(nvcc, 'Auto')
	gpu_dependencies = []
	gpu_dependencies += dependency('libdpdk', version : '>= 20.11.4.1.1')
	gpu_dependencies += dependency('cuda', version: '>= 11-5', modules: ['cudart', 'cuda'])
	# Refer to https://mesonbuild.com/Cuda-module.html
	add_project_arguments('-forward-unknown-to-host-compiler', language: 'cuda')
	if get_option('enable_cross_compilation_to_dpu')
                add_project_link_arguments(['-forward-unknown-to-host-linker', '--allow-shlib-undefined'], language : 'cuda')
        endif
endif

install_apps = false

base_cpp_args = [
	'-std=c++11'
]

# Comment this line to restore warnings of experimental DOCA features
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: languages)

if get_option('enable_cross_compilation_to_dpu')
	# Please update this folder if the base cross-compilation folder is located elsewhere
	add_project_arguments('-I/root/doca-cross/usr/include', language: languages)
endif

# Ensure mlnx-dpdk will manage to find our libbsd
add_project_arguments('-D RTE_USE_LIBBSD', language: languages)

base_app_dependencies = []
base_app_dependencies += dependency('threads')
base_app_dependencies += dependency('json-c')
base_app_dependencies += dependency('libbsd')

if get_option('enable_grpc_support')
	dependency_protobuf = dependency('protobuf')
	dependency_grpc = dependency('grpc++')

	protoc = find_program('protoc')
	grpc_cpp_plugin = find_program('grpc_cpp_plugin')
	grpc_py_plugin = find_program('grpc_python_plugin')
	# Hack to force the generation of non-executable generated files
	python_generator = find_program('cp', 'copy')

	grpc_app_dependencies = [dependency_protobuf, dependency_grpc]
endif

common_path = 'common/src'
common_dir_path = '../../' + common_path
common_grpc_dir_path = '../../' + common_dir_path + '/grpc'
base_app_inc_dirs = [include_directories(common_path)]

# Add the relevant GPU stuff to have a successful compilation with GPU support.
apps_gpu_common_src = []
if get_option('enable_gpu_support')
	base_app_dependencies += gpu_dependencies
	apps_gpu_common_src += files([
		common_path + '/gpu_init.c',
	])
else
	# With GPU support there is version limitation
	base_app_dependencies += dependency('libdpdk')
endif


source_path = 'src'
bin_path = 'bin'

app_list = [
	    'allreduce',
	    'application_recognition',
	    'dns_filter',
	    'file_scan',
	    'firewall',
	    'ips',
	    'secure_channel',
	    'simple_fwd_vnf',
	    'url_filter',
]

fs = import('fs')
if fs.is_dir('app_shield_agent')
	app_list += ['app_shield_agent']
endif

grpc_extended_apps = [
	'application_recognition',
	'dns_filter',
	'ips',
	'url_filter',
]

foreach APP_NAME : app_list
	if (not get_option('enable_' + APP_NAME) and not get_option('enable_all_applications'))
		warning('Skipping compilation of DOCA Application - @0@.'.format(APP_NAME))
		continue
	endif

	# gRPC preparation (just in case)
	app_supports_grpc = grpc_extended_apps.contains(APP_NAME) and get_option('enable_grpc_support')
	if app_supports_grpc
		common_grpc = '/'.join(['common', source_path, 'grpc'])
		common_proto = '/'.join([common_grpc, 'common.proto'])
		app_proto = APP_NAME + '.proto'

		# Create absolute grpc paths
		grpc_base_path   = '/'.join([APP_NAME, source_path, 'grpc'])
		grpc_server_path = '/'.join([grpc_base_path, 'server'])
		grpc_client_path = '/'.join([grpc_base_path, 'client'])

		bin_install_grpc_base_dir   = '/'.join([APP_NAME, bin_path, 'grpc'])
		bin_install_grpc_server_dir = '/'.join([bin_install_grpc_base_dir, 'server'])
		bin_install_grpc_client_dir = '/'.join([bin_install_grpc_base_dir, 'client'])

		# Generate the gRPC Server
		common_gen_cc = generator(protoc, \
					  output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
					  arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
						       '--cpp_out=@BUILD_DIR@',
						       '@INPUT@'])

		orchestration_gen_cc = generator(protoc, \
						 output    : ['@BASENAME@.grpc.pb.cc', '@BASENAME@.grpc.pb.h'],
						 arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
							      '--grpc_out=@BUILD_DIR@',
							      '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.full_path(),
							      '@INPUT@'])

		proto_gen_cc = generator(protoc, \
					 output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
					 arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						      '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						      '--cpp_out=@BUILD_DIR@',
						      '@INPUT@'])

		grpc_gen_cc = generator(protoc, \
					output    : ['@BASENAME@.grpc.pb.cc', '@BASENAME@.grpc.pb.h'],
					arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						     '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						     '--grpc_out=@BUILD_DIR@',
						     '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.full_path(),
						     '@INPUT@'])

		gen_common_cc = common_gen_cc.process(common_proto)
		gen_orchestration_cc = orchestration_gen_cc.process(common_proto)
		gen_proto_cc = proto_gen_cc.process(grpc_base_path + '/' + app_proto)
		gen_grpc_cc = grpc_gen_cc.process(grpc_base_path + '/' + app_proto)

		# Generate the gRPC Client
		common_gen_py = generator(protoc, \
					  output    : ['@BASENAME@_pb2.py'],
					  arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
						       '--python_out=@BUILD_DIR@',
						       '@INPUT@'])

		proto_gen_py = generator(protoc, \
					 output    : ['@BASENAME@_pb2.py'],
					 arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						      '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						      '--python_out=@BUILD_DIR@',
						      '@INPUT@'])

		grpc_gen_py = generator(protoc, \
					output    : ['@BASENAME@_pb2_grpc.py'],
					arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						     '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						     '--grpc_out=@BUILD_DIR@',
						     '--plugin=protoc-gen-grpc=' + grpc_py_plugin.full_path(),
						     '@INPUT@'])

		gen_common_py = common_gen_py.process(common_proto)
		gen_proto_py = proto_gen_py.process(grpc_base_path + '/' + app_proto)
		gen_grpc_py = grpc_gen_py.process(grpc_base_path + '/' + app_proto)
	endif

	# Basic app definitions
	app_dependencies = base_app_dependencies
	app_inc_dirs = base_app_inc_dirs
	app_libs = []
	app_srcs = apps_gpu_common_src

	# Enter the application's directory
	subdir(APP_NAME + '/' + source_path)

endforeach
