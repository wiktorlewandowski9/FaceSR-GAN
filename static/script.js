// Handles the file input change event, reads the selected image, and enables the start button.
document.getElementById('file').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                if (img.width === 16 && img.height === 16) {
                    document.querySelector('.input-image').src = e.target.result;
                    document.getElementById('start').disabled = false;
                } else {
                    alert('Please upload a 16x16 image.');
                    document.getElementById('file').value = '';
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Resets the file input and disables buttons on page load.
window.addEventListener('load', function() {
    document.getElementById('file').value = '';
    document.getElementById('start').disabled = true;
    document.getElementById('download').disabled = true;
});

// Sends the input image to the server for processing and displays the result.
document.getElementById('start').addEventListener('click', async function() {
    const imgSrc = document.querySelector('.input-image').src;
    if (imgSrc) {
        try {
            const response = await fetch('/process_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imgSrc })
            });
            const result = await response.json();
            if (result.processed_image) {
                document.querySelector('.output-image').src = result.processed_image;
                document.getElementById('download').disabled = false;
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }
});

// Downloads the processed image when the download button is clicked.
document.getElementById('download').addEventListener('click', function() {
    const outputImage = document.querySelector('.output-image').src;
    if (outputImage) {
        const link = document.createElement('a');
        link.href = outputImage;
        link.download = 'processed_image.png';
        link.click();
    }
});