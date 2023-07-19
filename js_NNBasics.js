async function tensorMathFucntions(){
    //adds an index by the corresponding index from the second matrix 
    let blue = tf.tensor([0,1,2]);
    const red = tf.tensor([2,4,6]);
    blue.add(red).print(); 


    //multiplies an index by the corresponding index from the second matrix, 
    //and then adds them together and returns the result
    blue = tf.tensor([0,1,2]);
    blue.dot(red).print(); 


    //Transpose flips the axis of the matrix
    blue = tf.tensor([[4,5],
                      [6,7],
                      [8,9]]).transpose().print();
}



async function tensorSetupTests(){

    const myTensor = tf.tensor([0,1,2]);
    const stringTensor = tf.tensor(["Hi","Hello"]);
    const twoD = tf.tensor2d([[0,1,2],[2, 4, 6]]);

    console.log("myTensor.rank: " + myTensor.rank);
    console.log("myTensor.shape: " + myTensor.shape);
    console.log("myTensor.dtype: " + myTensor.dtype);
    myTensor.print();
    stringTensor.print();
    twoD.print();

    //Reshpaes an array into smaller or larger arrays given the parameters
    //The product of the parameters must equal the size of the array
    tf.tensor([0,1,2,3,4,5])
    .reshape([1,3,2])
    .print(); 


    //Creates a tensor filled with ones
    tf.ones([4, 5]).print();

    //Creates a tensor filled with postives and negatives numbers
    tf.truncatedNormal([4, 5]).print();
}