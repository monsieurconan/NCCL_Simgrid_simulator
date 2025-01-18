#include "coll.h"
#include "cstdio"
#include "cuda.h"
#include "internal.h"
#include "nccl.h"
#include "nccl_internal.h"
#include "platform.h"
#include "smpi/mpi.h"
#include "util.h"
#include <iostream>

size_t start_size = 1e6;
size_t end_size = 1e8;

double inter_node_bw = 1e1;

COLL coll = COLL::SENDRECV;

int N_STEPS = 100;

int N_GPUS = 2;
int N_NODES = 1;

bool MPI = N_NODES >= 2;

int N_RANKS = N_GPUS * N_NODES;

namespace {
bool char_equal(const char a[], const char b[], int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void PRINT(size_t nb, int size, double time, double algbw, double busbw) {
    printf("    %10li     %10li   float  %8.2f   %6.2f   %6.2f\n", nb * size, nb, time * 1e6, algbw,
           busbw);
}

void print_actors() {
    auto all_actors = simgrid::s4u::Engine::get_instance()->get_all_actors();
    for (int i = 0; i < all_actors.size(); ++i) {
        std::cout << all_actors[i]->get_host()->get_name();
    }
}

void cleanup() {
    auto actors = simgrid::s4u::Engine::get_instance()->get_all_actors();
    for (int i = 0; i < actors.size(); ++i) {
        if (actors[i]->get_ppid() == simgrid::s4u::this_actor::get_pid()) actors[i]->kill();
    }
}

void test_all_gpu() {
    auto gpus =
        simgrid::s4u::Engine::get_instance()->get_filtered_hosts([](simgrid::s4u::Host *host) {
            if (char_equal(host->get_property("type"), "gpu", 3))
                return true;
            else
                return false;
        });
    for (int i = 0; i < gpus.size(); i++) {
        for (int j = 0; j < gpus.size(); j++)
            if (j != i) simgrid::s4u::Comm::sendto(gpus[j], gpus[i], 1000);
    }
}
} // namespace

template <typename F, typename... Args> void actor_wrapper(int user_rank, F code, Args... args) {
    // initialisation of actors extentions
    auto me = simgrid::s4u::Actor::self();
    me->extension_set<simgrid::cuda::cudaActor>(new simgrid::cuda::cudaActor(me));
    me->extension_set<ncclActor>(new ncclActor(user_rank));
    // actual code
    code(args...);
    // cleanup (killing the stream actors)
    cleanup();
}

template <typename F, typename... Args> void mpi_wrapper(F code, F dummy_code, Args... args) {
    MPI_Init();
    int user_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &user_rank);
    if (user_rank < N_NODES)
        actor_wrapper<F, Args...>(user_rank, code, args...);
    else {
        dummy_code(args...);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

void dummy_code_test(enum COLL coll_to_test, size_t nb_test, size_t start_size, int end_size) {
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<double> times(nb_test, 0);
    MPI_Reduce(times.data(), times.data(), times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

// fonction to change to your main
void nccl_test(enum COLL coll_to_test, size_t nb_test, size_t start_size, int end_size) {
    std::vector<cudaStream_t> streams(N_GPUS, nullptr);
    std::vector<ncclComm_t> comms(N_GPUS);
    int user_rank = nccl_actor()->u_rank;
    // initialisation of streams
    for (int i = 0; i < N_GPUS; ++i) {
        cudaSetDevice(i);
        cudaStreamCreateWithFlags(&streams[i], 0);
    }
    ncclUniqueId id = 0;
    if (user_rank == 0) ncclGetUniqueId(&id);
    // std::cout <<"streams initialised on "+simgrid::s4u::this_actor::get_name()+"\n";
    //  initialisation of buffers
    std::vector<void *> sendbufs(N_GPUS);
    std::vector<void *> recvbufs(N_GPUS);
    for (int i = 0; i < N_GPUS; ++i) {
        cudaMalloc(&sendbufs[i], end_size * sizeof(ncclFloat));
        cudaMalloc(&recvbufs[i], end_size * sizeof(ncclFloat));
    }
    // initialisation of comms
    ncclGroupStart();
    for (int i = 0; i < N_GPUS; ++i) {
        cudaSetDevice(i);
        ncclCommInitRank(&comms[i], N_RANKS, id, i + user_rank * N_GPUS);
    }
    ncclGroupEnd();
    if (MPI) MPI_Barrier(MPI_COMM_WORLD);
    // std::cout <<"comms initialised on "+simgrid::s4u::this_actor::get_name()+"\n";
    //  initialisation of graphs
    std::vector<cudaGraph_t> graphs(N_GPUS);
    std::vector<cudaGraphExec_t> graphExecs(N_GPUS);
    if (user_rank == 0) {
        std::cout << "          size          count    type      time    algbw    busbw\n";
        std::cout << "           (B)     (elements)              (us)   (GB/s)   (GB/s)\n";
    }
    std::vector<double> times;
    for (int iteration = 0; iteration < nb_test; iteration++) {
        // stream capture
        for (int i = 0; i < N_GPUS; ++i) {
            cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeThreadLocal);
        }
        size_t iteration_size = nb_test <= 1
                                    ? end_size
                                    : start_size/(nb_test-1) * (nb_test - 1 - iteration) +
                                          end_size/(nb_test-1) * (iteration);
        ncclGroupStart();
        for (int i = 0; i < N_GPUS; ++i) {
            switch (coll_to_test) {
            case REDUCE:
                ncclReduce(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, ncclSum, 0,
                           comms[i], streams[i]);
                break;
            case ALL_REDUCE:
                ncclAllReduce(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, ncclSum,
                              comms[i], streams[i]);
                break;
            case BROADCAST:
                ncclBcast(nullptr, iteration_size, ncclFloat, 0, comms[i], streams[i]);
                break;
            case REDUCE_SCATTER:
                ncclReduceScatter(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, ncclSum,
                                  comms[i], streams[i]);
                break;
            case ALL_GATHER:
                ncclAllGather(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, comms[i],
                              streams[i]);
                break;
            case GATHER:
                ncclGather(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, 0, comms[i],
                           streams[i]);
                break;
            case SCATTER:
                ncclScatter(sendbufs[i], recvbufs[i], iteration_size, ncclFloat, 0, comms[i],
                            streams[i]);
                break;
            case ALL_TO_ALL:
                ncclAllToAll(sendbufs[i], recvbufs[i], iteration_size, iteration_size, ncclFloat,
                             ncclFloat, comms[i], streams[i]);
                break;
            case SENDRECV:
                ncclSend(sendbufs[i], iteration_size, ncclFloat,
                         user_rank*N_GPUS + i == N_RANKS - 1 ? 0 : user_rank*N_GPUS + i+1, comms[i],
                         streams[i]);
                ncclRecv(recvbufs[i], iteration_size, ncclFloat,
                         user_rank*N_GPUS + i == 0 ? N_RANKS - 1 :  user_rank*N_GPUS + i- 1,
                         comms[i], streams[i]);
                break;
            case DEBUG:
                debugTest(streams[i]);
            default:
                break;
            }
        }
        ncclGroupEnd();
        for (int i = 0; i < N_GPUS; ++i) {
            cudaStreamEndCapture(streams[i], &graphs[i]);
            cudaGraphInstantiate(&graphExecs[i], graphs[i], nullptr, nullptr, 0);
        }
        // std::cout <<"graph initialised on "+simgrid::s4u::this_actor::get_name()+"\n";
        //  launching the kernels
        auto start_time = simgrid::s4u::Engine::get_instance()->get_clock();
        for (int i = 0; i < N_GPUS; ++i) {
            cudaGraphLaunch(graphExecs[i], streams[i]);
        }
        // std::cout <<"graph launched on "+simgrid::s4u::this_actor::get_name()+"\n";
        // streamsynchronisation
        for (int i = 0; i < N_GPUS; ++i) {
            cudaStreamSynchronise(streams[i]);
        }
        // std::cout <<"synchronised "+simgrid::s4u::this_actor::get_name()+"\n";
        auto end_time = simgrid::s4u::Engine::get_instance()->get_clock();
        times.push_back(end_time - start_time);
        for (int i = 0; i < N_GPUS; ++i) {
            cudaGraphDestroy(graphs[i]);
            cudaGraphExecDestroy(graphExecs[i]);
        }
    }
    if (MPI)
        MPI_Reduce(times.data(), times.data(), times.size(), MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
    if (user_rank == 0)
        for (int i = 0; i < nb_test; ++i) {
            size_t iteration_size = nb_test <= 1
                                    ? end_size
                                    : start_size/(nb_test-1) * (nb_test - 1 - i) +
                                          end_size/(nb_test-1) * (i);
            double avgtime = times[i];
            double alg_bw = getAlgoBandwidth(avgtime, iteration_size * sizeof(float));
            PRINT(iteration_size, 4, avgtime * 1e-6, alg_bw,
                  getBusBandwidth(coll_to_test, alg_bw, N_RANKS));
        }
    // todo check the operation
    // free the buffer
    /*for(int i=0;i<N_GPUS;++i){
        cudaFree(&sendbufs[i]);
        cudaFree(&recvbufs[i]);
    }*/
}

void nccl_test_mpi(enum COLL coll_to_test, int nb_test, int start_size, int end_size) {
    MPI_Init();
    int user_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &user_rank);
    //if (user_rank < N_NODES)
        actor_wrapper(user_rank, nccl_test, coll_to_test, nb_test, start_size, end_size);
    /*else {
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> times(nb_test, 0);
        MPI_Reduce(times.data(), times.data(), times.size(), MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    }*/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    simgrid::s4u::Engine engine("nccl");
    if(MPI) engine.set_config("smpi/simulate-computation:no");
    // engine.set_config("help:0");
    // engine.load_platform(argv[1]);
    create_starzone_default(N_NODES, inter_node_bw, N_GPUS);

    std::cout << "platform loaded\n";
    // getting all cpus
    auto cpus = engine.get_filtered_hosts([](simgrid::s4u::Host *host) {
        if (char_equal(host->get_property("type"), "cpu", 3))
            return true;
        else
            return false;
    });
    auto gpus = engine.get_filtered_hosts([](simgrid::s4u::Host *host) {
        if (char_equal(host->get_property("type"), "gpu", 3))
            return true;
        else
            return false;
    });
    for(int i=0;i<gpus.size();++i){
        // todo :  start the null stream ?
        gpus[i]->extension_set<ncclRank>(new ncclRank());
        
    }
    std::cout << cpus.size() << " cpu actors\n";
    std::cout << N_RANKS << " gpus \n";

    if (not MPI) {
        //for (int j = 0; j < cpus.size(); ++j)
        int j=0;
            auto actor = simgrid::s4u::Actor::create("cpu" + std::to_string(j), cpus[j], [&]() {
                actor_wrapper(j, nccl_test, coll, N_STEPS, start_size, end_size);
            });
    } else {
        SMPI_init();
        SMPI_app_instance_start(
            "nccl_test",
            []() { mpi_wrapper(nccl_test, dummy_code_test, coll, N_STEPS, start_size, end_size); },
            cpus);
    }

    engine.run();
    std::cout << "total time = " << engine.get_clock() << " s\n";
    if (MPI) SMPI_finalize();
    return 0;
}