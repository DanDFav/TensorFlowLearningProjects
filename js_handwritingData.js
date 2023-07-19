const numberOfDataSetSamples = 65000; 
const imageSize = 784;
const chunkSize = 5000; 
const imagePath = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const labelPath = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

class data {
    async load(){
        const image = new Image(); 
        const canvas = document.createElement("canvas");

        const context = canvas.getContext("2d");

        const imageRequest = new Promise ((resolve, reject) => {
            image.crossOrigin = ""; 
            image.onload = () => {
                bytesBuffer  = new ArrayBuffer(numberOfDataSetSamples * imageSize * 4); 
                for(let i = 0; i < numberOfDataSetSamples; i++){
                    const imageBytes = new Float32Array(bytesBuffer, i * imageSize * chunkSize * 4, imageSize * chunkSize); 
                    context.drawImage(image, 0, i * chunkSize, image.width, chunkSize, 0, 0, image.width, chunkSize)
                    const imageData = context.getImageData(0, 0, canvas.width, canvas.height); 

                    for(let imageDataIndex = 0; imageDataIndex < imageData.data.length / 4; imageDataIndex++){
                        imageBytes[imageDataIndex] = imageData.data[imageDataIndex * 4] / 255; 
                    }

                }
                this.datasetImages = new Float32Array(bytesBuffer); 
                resolve(); 
            }
            image.src = imagePath; 
        });

        const labelsRequest = fetch(labelPath);
        const [imageResponse, labelsResponse] = await Promise.all([imageRequest, labelsRequest]);

        this.labels = new Uint8Array(await labelsResponse.arrayBuffer());
    }
}