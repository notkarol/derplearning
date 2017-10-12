var vid = document.getElementById("front");
var fps = 1 / 20;
vid.ontimeupdate = function() { onTimeUpdate() };
function onTimeUpdate() {
    document.getElementById("demo").innerHTML = vid.currentTime;
}
function keyboard_callback(evt) {
    if (vid.paused) {
        if (evt.code == "Space") { vid.play(); }
        if (evt.code == "ArrowLeft") { vid.currentTime = Math.max(0, vid.currentTime - fps); }
        if (evt.code == "ArrowRight") { vid.currentTime = Math.min(vid.duration, vid.currentTime + fps); }
	if (evt.code == "ArrowDown") { vid.currentTime = Math.max(0, vid.currentTime - 1.0); }
        if (evt.code == "ArrowUp") { vid.currentTime = Math.min(vid.duration, vid.currentTime) + 1.0 }
    }
    else {
	if (evt.code == "Space") { vid.pause(); }
    }
}
window.addEventListener('keypress', keyboard_callback);
