# Off-Axis Digital Holography Processing

The code in the repository is highly "work-in-progress". In the future it is intended to optimize and use it for real-time processing of holograms.

## Dependencies
- opencv

## Usage
1. Clone the repository
2. Change the BASE_PATH variable in `main.cu` 
3. Run `make` to compile the code and then run it

## To-Do
- [x] Read and analyse hologram
- [ ] Use reference beam holograms to better reconstruct the object image
- [ ] Refactor code into OOP style
- [ ] Add timing code to measure performance
- [ ] Optionally optimize code for real-time processing (possibly by directly integrating CUDA for fast FFT).
