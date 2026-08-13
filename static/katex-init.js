document.addEventListener("DOMContentLoaded", function () {
	function renderProtectedMath(source, displayMode) {
		const wrapper = document.createElement("span");
		wrapper.className = displayMode ? "math-display" : "math-inline";

		try {
			katex.render(source, wrapper, {
				displayMode: displayMode,
				throwOnError: true,
			});
		} catch (error) {
			console.error("KaTeX protected-math render error:", error);
			return null;
		}

		return wrapper;
	}

	document.querySelectorAll("pre code[data-lang='math']").forEach(function (code) {
		const pre = code.closest("pre");
		const rendered = renderProtectedMath(code.textContent.trim(), true);

		if (pre && rendered) {
			pre.replaceWith(rendered);
		}
	});

	document.querySelectorAll("code").forEach(function (code) {
		if (code.closest("pre")) {
			return;
		}

		// GitHub-style inline math: $`...`$
		const previous = code.previousSibling;
		const next = code.nextSibling;
		const hasGitHubDelimiters = previous
			&& next
			&& previous.nodeType === Node.TEXT_NODE
			&& next.nodeType === Node.TEXT_NODE
			&& previous.textContent.endsWith("$")
			&& next.textContent.startsWith("$");

		if (hasGitHubDelimiters) {
			const rendered = renderProtectedMath(code.textContent, false);

			if (rendered) {
				previous.textContent = previous.textContent.slice(0, -1);
				next.textContent = next.textContent.slice(1);
				code.replaceWith(rendered);
			}
			return;
		}

		// Continue supporting the previous protected form for old or cached pages.
		const source = code.textContent.trim();
		const displayMode = source.startsWith("$$") && source.endsWith("$$");
		const inlineMode = source.startsWith("$") && source.endsWith("$");

		if (!displayMode && !inlineMode) {
			return;
		}

		const delimiterLength = displayMode ? 2 : 1;
		const expression = source.slice(delimiterLength, -delimiterLength);
		const rendered = renderProtectedMath(expression, displayMode);

		if (rendered) {
			code.replaceWith(rendered);
		}
	});

	renderMathInElement(document.body, {
		delimiters: [
			{ left: "$$", right: "$$", display: true },
			{ left: "$", right: "$", display: false },
		],
	});
});
