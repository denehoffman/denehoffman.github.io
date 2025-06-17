+++
+++
[`modak`](https://github.com/denehoffman/modak) is a simple-to-use, opinionated task queue system with dependency management, resource allocation, and isolation control. Tasks are run respecting topological dependencies, resource limits, and optional isolation.

This library only has two classes, `Task`s, which are an abstract class with a single method to override, `run(self) -> None`, and a `TaskQueue` which manages the execution order. Additionally, `modak` comes with a task monitor TUI which can be invoked with the `modak` shell command.

The `TaskQueue` has been written in Rust to get past issues with parallelism and the GIL. Instead of using a thread pool or even a multiprocessing pool, the tasks are serialized into bytes and passed to the Rust-side manager, which handles dispatching and execution. Each task is then run as a separate subprocess spawned in a Rust thread. This means the only way to share state between tasks is by writing to an output file and having a task depend on that file.

By default, `modak` scripts will create a state file called `.modak` in the current working directory. This can be changed by setting it in the `TaskQueue`'s initialization method. The `modak` CLI also supports an optional argument to point to the location of the state file.

