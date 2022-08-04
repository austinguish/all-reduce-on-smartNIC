import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
import geni.rspec as rspec
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()
# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()
# Node bf1
node_bf1 = request.RawPC('bf1')
node_bf1.hardware_type = 'r7525'
node_bf1.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD'
iface0 = node_bf1.addInterface('interface-0')
iface0.component_id = 'eth0'
iface0.addIPAddress(rspec.pg.IPv4Address("192.168.1.1", "255.255.255.0"))
iface1 = node_bf1.addInterface('interface-1')
iface1.addIPAddress(rspec.pg.IPv4Address("192.168.1.2", "255.255.255.0"))
# Node bf2
node_bf2 = request.RawPC('bf2')
node_bf2.hardware_type = 'r7525'
node_bf2.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD'
iface2 = node_bf2.addInterface('interface-2')
iface2.addIPAddress(rspec.pg.IPv4Address("192.168.2.1", "255.255.255.0"))
iface3 = node_bf2.addInterface('interface-3')
iface3.addIPAddress(rspec.pg.IPv4Address("192.168.2.2", "255.255.255.0"))
# Node bf3
node_bf3 = request.RawPC('bf3')
node_bf3.hardware_type = 'r7525'
node_bf3.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD'
iface4 = node_bf3.addInterface('interface-4')
iface4.addIPAddress(rspec.pg.IPv4Address("192.168.3.1", "255.255.255.0"))
iface5 = node_bf3.addInterface('interface-5')
iface5.addIPAddress(rspec.pg.IPv4Address("192.168.3.2", "255.255.255.0"))
# Link link-0
link_0 = request.Link('link-0')
link_0.Site('undefined')
iface4.bandwidth = 40000000
link_0.addInterface(iface2)
iface0.bandwidth = 40000000
link_0.addInterface(iface0)
# Link link-1
link_1 = request.Link('link-1')
link_1.Site('undefined')
iface1.bandwidth = 40000000
link_1.addInterface(iface1)
iface5.bandwidth = 40000000
link_1.addInterface(iface4)
# Link link-2
link_2 = request.Link('link-2')
link_2.Site('undefined')
iface2.bandwidth = 1000000
link_2.addInterface(iface3)
iface5.bandwidth = 1000000
link_2.addInterface(iface5)
# Print the generated rspec
pc.printRequestRSpec(request)
