<div align="center">
  <img src="https://github.com/windsornguyen/tenzor/blob/main/docs/static/tenzor.png" alt="Tenzor Logo">
</div>

--------------------------------------------------------------------------------

## Overview

Tenzor is a deep learning library written in Zig. It is currently in development (and is an excuse to learn Zig).

## TODO List

Some things that should be knocked out as soon as possible:

- [ ] Add full support for the most fundamental tensor operations
- [ ] Detect last checkpoint: WIP
- [ ] Autograd engine
- [ ] Basic linalg library

Some things that we'd like to add soon include GPU acceleration (see Karpathy's `llm.c`). Since Zig interoperates well with C/C++, I don't anticipate the `llm.c`-styled CUDA implementation to be overbearingly difficult.

## Installation

The library is currently being written in Zig 0.13.0.

## Usage

Still in development. You can try running `zig build run` in the root directory.

For more detailed usage instructions, please refer to the `docs/` directory. (TODO: To be made and document distributed training procedure)

## Limitations

Zig is still in its infancy, and there are a lot of things that are missing, e.g., complex numbers. Please bear with us and file issues if you find anything that's absurdly fundamental to be missing.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit issues, feature requests, and pull requests.

## License

Tenzor has a BSD-style license, as found in the [LICENSE](LICENSE) file.
