function analyzeStrip() {
    let fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        document.getElementById("result").innerText = "Please select an image";
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    document.getElementById("result").innerText = "Analyzing...";

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Status: " + data.status +
            "\nLines detected: " + data.lines_detected +
            "\nIntensities: " + data.intensities;
    })
    .catch(() => {
        document.getElementById("result").innerText = "Error occurred";
    });
}