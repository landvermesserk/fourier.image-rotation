# Build docker image
- docker build -t fourier-transform:0.1 -f docker/Dockerfile .

# Build docker container
- docker run -it -v $(pwd):/app --name fourier_transform fourier-transform:0.1 bash

# Create fourier transform
- python fourier_transform.py --image_path /app/samples/bagtag_0.png --pixel_threshold 250