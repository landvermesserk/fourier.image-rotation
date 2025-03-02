# Build docker image
- docker build -t fourier-transform:0.1 -f docker/Dockerfile .

# Build docker container
- docker run -it --name fourier_transform fourier-transform:0.1 bash