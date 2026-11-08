"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const { splitRanbooruStatus, updateProgressBar } = require("../javascript/ranbooru_progress.js");

assert.deepEqual(splitRanbooruStatus("Ranbooru: Processing 6003 results 3% ETA: 01:00:57"), {
    status: "Ranbooru: Processing 6003 results", progress: "3% ETA: 01:00:57",
});
assert.deepEqual(splitRanbooruStatus("Ranbooru: Searching yande.re..."), {
    status: "Ranbooru: Searching yande.re...", progress: "",
});
assert.equal(splitRanbooruStatus("15% ETA: 00:34"), null);
assert.equal(splitRanbooruStatus("ControlNet: running 50%"), null);

const children = [];
const fill = { textContent: "Ranbooru: Processing 6003 results 3% ETA: 01:00:57" };
const bar = {
    isConnected: true,
    querySelector(selector) {
        if (selector === ":scope > .progress") return fill;
        return children.find((child) => selector === ":scope > ." + child.className) || null;
    },
    appendChild(child) { children.push(child); child.remove = () => {
        const pos = children.indexOf(child);
        if (pos >= 0) children.splice(pos, 1);
    }; },
};
global.document = { createElement: () => ({ className: "", textContent: "" }) };
updateProgressBar(bar);
assert.equal(fill.textContent, "");
assert.equal(children.length, 2);
assert.equal(children[0].textContent, "Ranbooru: Processing 6003 results");
assert.equal(children[1].textContent, "3% ETA: 01:00:57");

// Forge updates the native progress element on every poll: keep overlays in sync.
fill.textContent = "Ranbooru: Processing 6003 results 4% ETA: 00:48:22";
updateProgressBar(bar);
assert.equal(children.length, 2);
assert.equal(children[1].textContent, "4% ETA: 00:48:22");
assert.equal(fill.textContent, "");

// If the result summary is cleared, keep the last status until the bar is removed.
fill.textContent = "5% ETA: 00:44:00";
updateProgressBar(bar);
assert.equal(children[0].textContent, "Ranbooru: Processing 6003 results");
assert.equal(children[1].textContent, "5% ETA: 00:44:00");

// Don't move or overwrite another extension's message.
fill.textContent = "ControlNet: loading models 6% ETA: 00:43:00";
updateProgressBar(bar);
assert.equal(children.length, 0);
assert.equal(fill.textContent, "ControlNet: loading models 6% ETA: 00:43:00");

console.log("Ranbooru progress overlay tests passed");
