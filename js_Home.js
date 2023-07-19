const IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];



if(document.readyState != "loading"){
    run();
} else {
    document.addEventListener("DOMContentLoaded", run);
}

async function run(){
    console.log("Setup some buttons later.");

    speechRecogMain(); 
}


