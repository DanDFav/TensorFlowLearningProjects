let modelSR; 

const possibleDirections = ["right", "left", "up", "down"];
const possibleNumbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

async function speechRecogMain(){
    modelSR = speechCommands.create("BROWSER_FFT"); 

    await modelSR.ensureModelLoaded();

    console.log("Model Loaded"); 
    console.log(modelSR); 

    const canvas = document.getElementById("drawingCanvas");
    const context = canvas.getContext("2d");

    context.lineWidth = 15; 
    context.lineHeight = 15; 
    context.strokeStyle = "black";

    var xPos = 250; 
    var yPos = 250;

    const detectedWords = modelSR.wordLabels();

    modelSR.listen(({scores}) => {
        scores = Array.from(scores).map((s, index) => ({
            score: s, word: detectedWords[index]
        }));

        scores.sort((s1, s2) => s2.score - s1.score);

        var detectedWord = scores[0].word;
        console.log(detectedWord);
        if(possibleDirections.includes(detectedWord)){
            [xPos, yPos] = draw(context, xPos, yPos, detectedWord);
        } else if(possibleNumbers.includes(detectedWord)){
            context.strokeStyle = newColour(detectedWord); 
        }
    }, {probabilityThreshold: 0.95});
}


function draw(context, xPos, yPos, directionOfDrawing) {
    var [newXPos, newYPos] = findNewPos(xPos, yPos, directionOfDrawing);

    context.beginPath();
    context.moveTo(xPos, yPos);
    context.lineTo(newXPos, newYPos);
    context.closePath();
    context.stroke(); 

    xPos = newXPos; 
    yPos = newYPos;    

    return [xPos, yPos]
}

function newColour(colour){
    return {
        "zero": "black",
        "one": "red",
        "two": "blue",
        "three": "green",
    }[colour]
}

function findNewPos(prevX, prevY, direction){
    return {
        "left": [prevX - 15, prevY],
        "right": [prevX + 15, prevY], 
        "up": [prevX, prevY - 15], 
        "down": [prevX, prevY + 15], 
        "default": [prevX, prevY]
    }[direction]
}