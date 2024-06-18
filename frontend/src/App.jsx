import { useState } from 'react'
import './App.css'
import axios from 'axios'

function App() {
  // Defining state variables. 'useState' initalizes the layers state variable to 3.
  // Then, it provides a function 'setLayers' to update it. Similar initialization for the other state
  // variables.
  const [layers, setLayers] = useState(3);
  const [units, setUnits] = useState([32, 64, 128]);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [optimizer, setOptimizer] = useState('adam');
  const [accuracy, setAccuracy] = useState(null);
  const [loss, setLoss] = useState(null);

  // Here we're defining an asynchronous function named handleTrain using arrow function syntax.
  // This function takes an event object 'e' as a parameter, which is typically the event triggered
  // by a form submission or button click. Keyword 'async' indicates that this function is asynchronous
  // and will return a promise.
  const handleTrain = async(e) => {
    // Preventing the default form submission behavior. That is, if 'e' is a form submission event,
    // calling 'preventDefault()' will prevent the form from being submitted. This means we can handle
    // the form submission or button click programmatically ourselves via javascript.
    e.preventDefault();

    // Making a POST request to the backend server using axios. The POST request is made to the
    // '/train' endpoint of the backend server. The way we have Flask backend setup is so that it runs
    // on local machine IP '127.0.0.1' and port 5000. 
    // The second argument to the 'post' method is the data we want to send to the server. This data
    // is an object with keys 'layers', 'units', 'epochs', 'batchSize', and 'optimizer'. The values of
    // these keys are the state variables defined above.
    const response = await axios.post('http://localhost:5000/train', {
      layers: layers,
      units: units,
      epochs: epochs,
      batchSize: batchSize,
      optimizer: optimizer
    });

    // Using the aforementioned 'setAccuracy' and 'setLoss' functions to update the accuracy and loss
    // with data from the response object
    setAccuracy(response.data.accuracy);
    setLoss(response.data.loss);
  };
  return (
    <div className="App">
        <h1>CIFAR-10 Model Training UI</h1>

        {/* Form element with 'onSubmit' event handler set to the 'handleTrain' function we defined 
        earlier. 
        This means that when the form is submitted, the 'handleTrain' function will be called.
        */}
        <form onSubmit={handleTrain}>
            <div>
                <label>Number of Layers: </label>
                {/* Defining an input field for entering number of layers. 
                'value={layers}' sets the initial value of the input field to the current value of 
                the 'layers' state variable.
                'onChange={(e) => setLayers(parseInt(e.target.value))}' updates the 'layers' state
                when the input field value changes. */}
                <input type="number" value={layers} onChange={(e) => setLayers(parseInt(e.target.value))} />
            </div>
            {/* Commenting out control over node count for now. Will return to this later. */}
            {/* <div>
                <label>Units in Layers (comma-separated): </label>
                <input
                    type="text"
                    value={units.join(',')}
                    onChange={(e) => setUnits(e.target.value.split(',').map(Number))}
                />
            </div> */}
            <div>
                <label>Number of Epochs: </label>
                <input type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} />
            </div>
            <div>
                <label>Batch Size: </label>
                <input type="number" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))} />
            </div>
            <div>
                <label>Optimizer: </label>
                <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
                    <option value="adam">Adam</option>
                    <option value="sgd">SGD</option>
                    <option value="rmsprop">RMSprop</option>
                </select>
            </div>
            <button type="submit">Train Model</button>
        </form>
        {accuracy !== null && (
            <div>
                <h2>Model Results</h2>
                <p>Test Accuracy: {accuracy}</p>
                <p>Test Loss: {loss}</p>
            </div>
        )}
    </div>
);
}

export default App
