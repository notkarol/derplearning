var vid = document.getElementById("front");
var fps = 1 / 20;
var n_frames = Math.round(vid.duration / fps + 0.5);
var write_state = "";
vid.ontimeupdate = function() { onTimeUpdate() };
function onTimeUpdate() {
    var out = "";
    var frame_num = Math.round(vid.currentTime / fps + 0.5);
    out = out.concat(frame_num, " / ", n_frames);
    document.getElementById("time").innerHTML = out;
}
function keyboard_callback(evt) {
    if (evt.code == "g") { write_state = "good"; }
    if (evt.code == "b") { write_state = "bad"; }
    if (evt.code == "t") { write_state = "trash"; }
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
