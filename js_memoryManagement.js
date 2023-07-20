function mmMain(){
    let a, b, c, d, e; 

    console.log("Inintial Tensors: ", tf.memory().numTensors);

    tf.tidy(()=>{
        a = tf.tensor([0,1]);
        b = tf.tensor([2,3]);
        c = tf.tensor([4,5]);
        d = tf.tensor([6,7]);

        tf.keep(d); //Keeps d 
        
        console.log("After tensor creation: " + tf.memory().numTensors);
        return c; //keeps c 
    });

    console.log("After tensor tidy: " + tf.memory().numTensors);

    d.dispose(); 

    console.log("After tensor dispose: " + tf.memory().numTensors);
}