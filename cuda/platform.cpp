#include "platform.h"

double cpu_speed = 1e3;
double latency = 1e-6;
//simgrid::xbt::Extension<simgrid::s4u::Host, cudaDeviceProp> cudaDeviceProp::EXTENSION_ID = simgrid::s4u::Host::extension_create<cudaDeviceProp>();

cudaDeviceProp::cudaDeviceProp()
{
    clockRate = 1e3;
    maxThreadsPerBlock = 256;
    concurrentKernels = 4;
}

double cudaDeviceProp::get_speed()
{
    return 1e12;
}

int cudaDeviceProp::parallelisation_degree()
{
    return concurrentKernels;
}


simgrid::s4u::NetZone *create_simple_node(simgrid::s4u::NetZone *root, int i_node, int ncpus_cores, int ngpus, cudaDeviceProp gpu_prop, double cpu_to_gpu_bandwidth, double gpu_to_gpu_bandwidth)
{
    auto node = simgrid::s4u::create_dijkstra_zone("node"+std::to_string(i_node), true);
    node->set_parent(root);
    auto router = node->create_router("irouter"+std::to_string(i_node));
    node->set_gateway(router);
    auto routerlink = node->create_link("ir"+std::to_string(i_node), gpu_to_gpu_bandwidth)->set_latency(latency);
    simgrid::s4u::LinkInRoute routerlink_in_route{routerlink};
    std::vector<simgrid::s4u::Host*> cpu_cores;
    std::vector<simgrid::s4u::Host*> gpus;
    for(int i=0;i<ncpus_cores;++i){
        cpu_cores.push_back(node->create_host("n"+std::to_string(i_node)+"cpu"+std::to_string(i), cpu_speed));
        cpu_cores[i]->set_property("type", "cpu");
        node->add_route(cpu_cores[i]->get_netpoint(), router, nullptr, nullptr,{routerlink_in_route});
    }
    for(int i=0;i<ngpus;++i){
        gpus.push_back(node->create_host("n"+std::to_string(i_node)+"_gpu"+std::to_string(i), gpu_prop.get_speed()));
        gpus[i]->set_core_count(gpu_prop.parallelisation_degree());
        gpus[i]->set_property("type", "gpu");
        //gpus[i]->extension_set<cudaDeviceProp>(&gpu_prop);
        node->add_route(gpus[i]->get_netpoint(), router, nullptr, nullptr, {routerlink_in_route});
        for(int j=0;j<ncpus_cores;++j){
            auto link = node->create_link("in"+std::to_string(i_node)+"cpu"+std::to_string(j)+":"+std::to_string(i), cpu_to_gpu_bandwidth)->set_latency(latency);
            simgrid::s4u::LinkInRoute link_in_route{link};
            node->add_route(cpu_cores[j], gpus[i], {link_in_route}, true);
        }
        for(int j=0;j<i;++j){
            auto link = node->create_link("in"+std::to_string(i_node)+"gpu"+std::to_string(j)+":"+std::to_string(i), gpu_to_gpu_bandwidth)->set_latency(latency);
            simgrid::s4u::LinkInRoute link_in_route{link};
            node->add_route(gpus[j], gpus[i], {link_in_route}, true);
        }
    }
    node->seal();
    return node;
}

void create_dummy_host(simgrid::s4u::NetZone *root){
    auto node = simgrid::s4u::create_full_zone("zzz");
    node->set_parent(root);
    auto router = node->create_router("zrouter");
    node->set_gateway(router);
    auto l = node->create_link("zlink", 1e9);
    auto h = node->create_host(host_no_deadlock, 1)->set_property("type", "not_really");
    node->add_route(h->get_netpoint(), router, nullptr, nullptr, {simgrid::s4u::LinkInRoute{l}});
    node->seal();
    auto link = root->create_link("special_link", 1e9);
    simgrid::s4u::LinkInRoute link_in_route{link};
    root->add_route(node, nullptr, {link_in_route});
}

void create_starzone_default(int nspikes, double inter_node_bandwidth, int ngpu_per_core)
{
        auto root = simgrid::s4u::create_star_zone("root");
        auto router = root->create_router("orouter");
        root->set_gateway(router);
        auto backbone = root->create_link("backbone", inter_node_bandwidth)->set_latency(latency);
        simgrid::s4u::LinkInRoute link_in_route{backbone};
        auto device_props = cudaDeviceProp();
        for(int i=0;i<nspikes;++i){
            auto node = create_simple_node(root, i,1, ngpu_per_core, device_props, 1e9, 1.2e10);
            node->seal();
            
            root->add_route(node, nullptr, {link_in_route});
        }
        create_dummy_host(root);
        root->seal();
}