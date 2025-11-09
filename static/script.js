const main = document.getElementById("main");
const recognize = document.getElementById("recognize");
const registerDiv = document.getElementById("register");

document.getElementById("recognizeBtn").onclick = () => {
    main.classList.add("hidden");
    recognize.classList.remove("hidden");
};

document.getElementById("registerBtn").onclick = () => {
    main.classList.add("hidden");
    registerDiv.classList.remove("hidden");
};

document.getElementById("back1").onclick = () => {
    recognize.classList.add("hidden");
    main.classList.remove("hidden");
};

document.getElementById("back2").onclick = () => {
    registerDiv.classList.add("hidden");
    main.classList.remove("hidden");
};

document.getElementById("registerSubmit").onclick = async () => {
    const name = document.getElementById("name").value;
    const file = document.getElementById("registerFile").files[0];
    if (!name || !file) return alert("Please enter name and upload image");

    const formData = new FormData();
    formData.append("name", name);
    formData.append("image", file);

    const res = await fetch("/register", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("registerResult").textContent = data.message;
};

document.getElementById("recognizeFile").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("image", file);
    const res = await fetch("/recognize", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("recognizeResult").textContent =
        data.status === "success" ? `Welcome ${data.name}!` : "Unknown Face";
};
