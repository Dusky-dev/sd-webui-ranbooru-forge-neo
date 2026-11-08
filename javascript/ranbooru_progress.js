/* Ranbooru status belongs on the left of Forge Neo's progressDiv, while
 * percentage/ETA remains on the right.  Forge Neo rebuilds the bar on every
 * generation and replaces .progress text about twice per second.  Observe
 * those changes rather than patching Forge Neo's requestProgress function.
 */
(function () {
    "use strict";

    const STATUS_CLASS = "ranbooru-progress-status";
    const DETAIL_CLASS = "ranbooru-progress-detail";

    function splitRanbooruStatus(text) {
        const match = /^Ranbooru:\s*(Searching\s+.+?\.{3}|Processing\s+[\d,.]+\s+results)(?:\s+(.*))?$/.exec(text || "");
        if (!match) return null;
        return {
            status: "Ranbooru: " + match[1],
            progress: match[2] || "",
        };
    }

    function updateProgressBar(bar) {
        if (!bar || !bar.isConnected) return;
        const fill = bar.querySelector(":scope > .progress");
        if (!fill) return;

        const text = (fill.textContent || "").trim();
        if (!text) return;

        const parsed = splitRanbooruStatus(text);
        let status = bar.querySelector(":scope > ." + STATUS_CLASS);
        let detail = bar.querySelector(":scope > ." + DETAIL_CLASS);
        if (!parsed && !status) return;

        // Leave another extension's status alone if it replaces Ranbooru's.
        if (!parsed && !/^(?:\d+(?:\.\d+)?%|ETA:)/.test(text)) {
            status?.remove();
            detail?.remove();
            return;
        }

        if (!status) {
            status = document.createElement("span");
            status.className = STATUS_CLASS;
            bar.appendChild(status);
        }
        if (!detail) {
            detail = document.createElement("span");
            detail.className = DETAIL_CLASS;
            bar.appendChild(detail);
        }
        if (parsed && status.textContent !== parsed.status) {
            status.textContent = parsed.status;
        }
        const progress = parsed ? parsed.progress : text;
        if (detail.textContent !== progress) {
            detail.textContent = progress;
        }

        // The .progress element remains the filling bar; only its text moves.
        // The next Forge Neo progress poll will write new text to this element.
        fill.textContent = "";
    }

    function installProgressObserver() {
        const root = typeof gradioApp === "function" ? gradioApp() : document;
        if (!root || root.__ranbooruProgressObserver) return;

        const observer = new MutationObserver((mutations) => {
            const bars = new Set();
            for (const mutation of mutations) {
                const target = mutation.target.nodeType === 1
                    ? mutation.target
                    : mutation.target.parentElement;
                const enclosing = target?.closest?.(".progressDiv");
                if (enclosing) bars.add(enclosing);

                for (const node of mutation.addedNodes || []) {
                    if (node.nodeType !== 1) continue;
                    if (node.matches(".progressDiv")) bars.add(node);
                    node.querySelectorAll?.(".progressDiv").forEach((bar) => bars.add(bar));
                }
            }
            bars.forEach(updateProgressBar);
        });
        observer.observe(root, { childList: true, characterData: true, subtree: true });
        root.__ranbooruProgressObserver = observer;
        root.querySelectorAll(".progressDiv").forEach(updateProgressBar);
    }

    if (typeof module !== "undefined" && module.exports) {
        module.exports = { splitRanbooruStatus, updateProgressBar };
    }
    if (typeof document === "undefined" || typeof MutationObserver === "undefined") return;

    if (typeof onUiLoaded === "function") {
        onUiLoaded(installProgressObserver);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", installProgressObserver, { once: true });
    } else {
        installProgressObserver();
    }
})();
