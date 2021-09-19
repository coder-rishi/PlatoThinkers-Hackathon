// listener to report.py

const { spawn } = require('child_process');
const bhools = []; // Store readings

const sensor = spawn('python', ['report.py']);

sensor.stdout.on('data', function(data) {

    

    // Coerce Buffer object to Float

    bhools.push(parseFloat(data));

    

    if (bhools.toString() == 0) {
        fin = 'True'
    } else {
        fin = 'False'
    }



    // Log to debug
    console.log(fin.toString());

});