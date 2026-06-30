const socket = io({
    transports:["websocket"]
})
const terminal = document.querySelector(".terminal");
const bar = document.querySelector(".progress-bar");

socket.emit("ready");

socket.on("log", (msg) =>{
    console.log(msg)
    terminal.textContent += msg + "\n";
    terminal.scrollTop = terminal.scrollHeight;
})

socket.on("progress", (percent)=>{
    bar.style.width = percent + "%";
})

socket.on("done", (data)=> {
    terminal.textContent += "\n" + data.message + "\n";
})