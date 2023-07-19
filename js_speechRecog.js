let modelSR; 

async function speechRecogMain(){
    modelSR = speechCommands.create("BROWSER_FFT", "directional4w"); 

    await modelSR.ensureModelLoaded();

    console.log("Model Loaded"); 

    const canvas = document.getElementById("drawingCanvas");
    const context = canvas.getContext("2d");

    context.lineWidth = 5; 

    var xPos = 250; 
    var yPos = 250; 

    draw(context, xPos, yPos);
}


function draw(context, xPos, yPos) {
    const detectedWords = modelSR.wordLabels();
    
    modelSR.listen(({scores}) => {
        scores = Array.from(scores).map((s, index) => ({
            score: s, word: detectedWords[index]
        }));

        scores.sort((s1, s2) => s2.score - s1.score);

        var directionOfDrawing = scores[0].word;
        console.log(directionOfDrawing);

        var [newXPos, newYPos] = findNewPos(xPos, yPos, directionOfDrawing);

        context.moveTo(xPos, yPos);
        context.lineTo(newXPos, newYPos);
        context.closePath();
        context.stroke(); 

        xPos = newXPos; 
        yPos = newYPos;
    });
}

function findNewPos(prevX, prevY, direction){
    return {
        "left": [prevX - 5, prevY],
        "right": [prevX + 5, prevY], 
        "up": [prevX, prevY - 5], 
        "down": [prevX, prevY + 5], 
        "default": [prevX, prevY]
    }[direction]
}