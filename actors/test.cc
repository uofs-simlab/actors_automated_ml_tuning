#include "config.h" // Includes our config class, message atoms, and CAF

// Define the job structure: one job = one beta value to train + evaluate
struct job_t {
    double beta;
    // Serialization for CAF
    template <class Inspector>
    friend bool inspect(Inspector& f, job_t& j) {
        return f.object(j).fields(
            f.field("beta", j.beta)
        );
    }
};
// Register job_t with CAF
CAF_BEGIN_TYPE_ID_BLOCK(my_project, caf::first_custom_type_id)
CAF_ADD_TYPE_ID(my_project, (job_t))
CAF_END_TYPE_ID_BLOCK(my_project)

// Worker state and behavior
struct worker_state {
    actor manager_actor;
};

behavior worker_actor(stateful_actor<worker_state>* self, actor manager, int num_gpus) {
    self->println("Worker actor is spawned successfully!");
    self->state().manager_actor = manager;

    return {
        [=](job_t job) {
            // popen()/pclose() block until the python training run finishes, which can
            // take minutes. Running that inline here would block this actor's mailbox
            // (and, if this actor isn't detached, a shared CAF scheduler thread too).  
            // Instead, hand the blocking call off to its own std::thread and return
            // immediately, so the worker actor stays fully responsive while it works.
            actor self_hdl{self};       // owning handle so the actor survives until the thread reports back
            actor mgr = self->state().manager_actor;

            // Shared across every worker_actor in this process: each job that gets
            // dispatched claims the next GPU index, round-robin, so concurrent training
            // runs spread across all local GPUs instead of piling onto GPU 0.
            static std::atomic<int> next_gpu{0};
            int gpu_id = num_gpus > 0 ? (next_gpu.fetch_add(1) % num_gpus) : -1;

            std::thread([self_hdl, mgr, job, gpu_id]() mutable {
                double result = 0.0;

                // --quiet: latentflow.py sends its training logs to stderr and prints
                // only the final MMD score to stdout, which is all we parse below.
                // Use the project .venv's python directly so this doesn't depend on
                // whichever python3 happens to be first on PATH (that's how the
                // torch/torchvision/etc. dependencies actually get found).
                std::ostringstream command;
                if (gpu_id >= 0) {
                    // latentflow.py just does `"cuda" if torch.cuda.is_available()`, so it
                    // always grabs whatever CUDA sees as device 0. Restricting visibility to
                    // one physical GPU per process is what actually pins this run to it.
                    command << "CUDA_VISIBLE_DEVICES=" << gpu_id << " ";
                }
                command << "../.venv/bin/python3 ./latentflow1.py --beta " << job.beta << " --quiet";

                FILE* pipe = popen(command.str().c_str(), "r");
                if (!pipe) {
                    std::cerr << "Failed to execute Python script for beta=" << job.beta << "\n";
                    anon_mail("result", job.beta, result, self_hdl).send(mgr);
                    return;
                }

                char buffer[128];
                std::string output;
                while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                    output += buffer;
                }

                int return_code = pclose(pipe);
                if (return_code == 0) {
                    try {
                        result = std::stod(output);
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing result for beta=" << job.beta << ": " << e.what() << "\n";
                    }
                } else {
                    std::cerr << "Python script failed for beta=" << job.beta
                               << " with return code: " << return_code << "\n";
                }

                // Thread-safe: actor mailboxes can be posted to from any thread.
                anon_mail("result", job.beta, result, self_hdl).send(mgr);
            }).detach();
        },
        [=](std::string message) {
            if (message == "quit") {
                self->quit();
            }
        },
    };
}
struct remote_state
{
    caf::actor manager;
};

caf::behavior remote(caf::stateful_actor<remote_state> *self, std::string hostname, uint16_t port, int num_gpus)
{

    // The below code connects this actor to a published actor on a remote host
    auto server_actor = self->system().middleman().remote_actor(hostname, port);
    if (!server_actor)
    {
        self->println("Failed to connect to remote actor at " + hostname + ":" + to_string(port));
        self->quit();
        return {}; // exits the behavior without running the below code
    }
    else
    {
        self->println("Successfully connected to remote actor at " + hostname + ":" + to_string(port));
    }

    self->state().manager = *server_actor;

    // send the refrence to manager
    anon_mail(self, 0).send(self->state().manager);

    return {
        [=](int actorNumber)
        {
            for (int i = 0; i < actorNumber; ++i)
            {
                // Not detached: the handler below never blocks (it hands the
                // python call off to its own std::thread), so these can share
                // CAF's normal scheduler pool.
                actor worker = self->spawn(worker_actor, self->state().manager, num_gpus);
                self->println("Spawning worker actor on {} on remote", i);
                anon_mail(worker).send(self->state().manager);
            }
            self->quit();
        }
    };
}


// Server state and behavior
struct server_state {
    std::queue<job_t> jobs;
    int actor_number = 3;
    std::chrono::high_resolution_clock::time_point start_time;
    int jobs_count = 0;
    int jobs_completed = 0;
};

behavior server(stateful_actor<server_state>* self, uint16_t port, std::vector<double> betas, int num_workers, int num_gpus) {
    // Start the timer
    self->state().start_time = std::chrono::high_resolution_clock::now();
    self->state().actor_number = num_workers;
    auto is_port = self->system().middleman().publish(self, port);
    if (!is_port) {
        self->println("Failed to publish actor on port " + to_string(port));
        self->quit();
    }
    else {
        self->println("Server is running on port " + to_string(port));
    }

    // One job per beta value in the sweep
    for (double beta : betas) {
        self->state().jobs.push({beta});
    }
    self->state().jobs_count = self->state().jobs.size();
    self->println("jobs are created successfully with {} ",self->state().jobs.size());

        // Spawn worker actors
        for (int i = 0; i < self->state().actor_number; ++i) {
            self->println("Spawning worker actor {} on server", i);
            // Not detached: the handler never blocks (the python call runs on
            // its own std::thread), so workers can share CAF's scheduler pool.
            actor worker = self->spawn(worker_actor, self, num_gpus);
            if (!self->state().jobs.empty())
            {
                auto job = self->state().jobs.front();
                self->state().jobs.pop();
                anon_mail(job).send(worker);
                self->println("Number of remaining jobs are: {}", self->state().jobs.size());
            }
            else
            {
                self->println("No more jobs to send to worker actor");
                anon_mail("quit").send(worker);
            }
        }

     return {
        [=](std::string msg, double beta, double result, actor worker_actor) {
            if (msg == "result") {
                self->println("Worker completed: beta = {}, MMD = {}", beta, result);
                self->state().jobs_completed++;
                if (!self->state().jobs.empty()) {
                    auto job = self->state().jobs.front();
                    self->state().jobs.pop();
                    anon_mail(job).send(worker_actor);
                    self->println("Number of remaining jobs are: {}", self->state().jobs.size());
                }
                else{
                    anon_mail("quit").send(worker_actor);
                    if (self->state().jobs_completed == self->state().jobs_count)
                    {
                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - self->state().start_time);
                        self->println("All jobs are completed in {} seconds", duration.count());
                        self->quit();
                    }
                }
            }
        },
        [=](actor remote, int none)
        {
            // self->println("Remote actor has requested for actor number");
            // send the refrence to manager
            anon_mail(self->state().actor_number).send(remote);
        },
        [=](actor worker)
        {
            if (!self->state().jobs.empty())
            {
                auto job = self->state().jobs.front();
                self->state().jobs.pop();
                anon_mail(job).send(worker);
                self->println("Number of remaining jobs are: {}", self->state().jobs.size());
            }
            else
            {
                anon_mail("quit").send(worker);
            }
        },
    };


}

// Main entry point for CAF
void caf_main(actor_system& system, const config& cfg) {

    caf::scoped_actor self{system};
    
    self->println("Port: " + to_string(cfg.port) + " Host: " + cfg.host);
    self->println("Host: " + cfg.host);
    self->println("Server Mode: " + to_string(cfg.server_mode));

    if (cfg.server_mode)
    {
        auto server_actor = system.spawn(server, cfg.port, parse_betas(cfg.betas), cfg.workers, cfg.gpus);
    }
    else
    {
        auto remote_actor = system.spawn(remote, cfg.host, cfg.port, cfg.gpus);
    }
}
CAF_MAIN(io::middleman, caf::id_block::my_project)