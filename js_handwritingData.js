const numberOfDataSetSamples = 65000; 
const imageSize = 784;
const chunkSize = 5000; 
const imagePath = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const labelPath = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";
const numberOfTrainingSamples = numberOfDataSetSamples * 0.8;  
const numberOfClasses = 10; 

class data {

    constructor(){

    }

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

        this.trainX = this.datasetImages.slice(0, imageSize * numberOfTrainingSamples); 
        this.testX = this.datasetImages.slice(imageSize * numberOfTrainingSamples);

        this.trainY = this.labels.slice(0, numberOfClasses * numberOfTrainingSamples);
        this.testY = this.labels.slice(numberOfClasses * numberOfTrainingSamples);  
    }

    getTrainingBatch(batchSize){
        return this.getNextBatch(batchSize, data, index);
    }

    getNextBatch(batchSize, data, index) {
        const imagesArray = new Float32Array(batchSize * imageSize); 
        const labelsArray = new Uint8Array(batchSize * numberOfClasses); 
        for(let i = 0; i < batchSize; i++){
            const currentIndex = index; 
            const image = data[0].slice(currentIndex * imageSize, currentIndex * imageSize + imageSize); 

            imagesArray.set(image, i * imageSize);

            const label = data[1].slice(currentIndex * numberOfClasses, currentIndex * numberOfClasses + numberOfClasses)
            labelsArray.set(label, i * numberOfClasses); 

            const Xtensor = tf.tensor2d(imagesArray, [batchSize, imageSize]);
            const Ytensor = tf.tensor2d(labelsArray, [batchSize, numberOfClasses]);

            return {Xtensor, Ytensor};
        }

    }
}