const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

let selectedFile;

// Clear previous result when a new file is selected
fileInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    preview.innerHTML = "";
    result.className = ""; // Clear previous result color
    result.innerHTML = ""; // Clear previous result text

    if (selectedFile) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(selectedFile);
        preview.appendChild(img);
    }
});

predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please select an image first!");
        return;
    }

    // Show loading text
    result.className = "loading";
    result.textContent = "Predicting... ⏳";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        // Check if response is OK
        if (!response.ok) {
            const text = await response.text(); // get raw response
            throw new Error(`Server error: ${response.status}\n${text}`);
        }

        const data = await response.json();

        if (data.error) {
            result.className = "result-pneumonia";
            result.textContent = data.error;
        } else {
            const resultClass = data.prediction === "NORMAL" ? "result-normal" : "result-pneumonia";
            result.className = resultClass;
            result.innerHTML = `Prediction: ${data.prediction} <br> ${data.message}`;
        }
    } catch (err) {
        result.className = "result-pneumonia";
        result.textContent = "Prediction failed. Check console.";
        console.error("Fetch error:", err);
    }
});
