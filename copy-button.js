// Based on https://www.roboleary.net/2022/01/13/copy-code-to-clipboard-blog.html
document.addEventListener("DOMContentLoaded", function () {
	// Support both Zola 0.21 (pre[class^='language-']) and 0.22 (pre.giallo).
	// Math fences are rendered as equations, so they should not get code headers.
	let zola22Blocks = Array.from(document.querySelectorAll("pre.giallo"));
	let zola21Blocks = Array.from(document.querySelectorAll("pre[class^='language-']"));
	let blocks = [...zola22Blocks, ...zola21Blocks].filter((block) => {
		let codeElement = block.querySelector("code");
		let lang = (codeElement && codeElement.getAttribute("data-lang"))
			|| block.getAttribute("data-lang");
		return !block.closest("div.crt") && lang !== "math";
	});

	blocks.forEach((block) => {
		if (navigator.clipboard) {
			let title = document.createElement("span");
			let codeElement = block.querySelector("code");
			let lang = (codeElement && (codeElement.getAttribute("data-name") || codeElement.getAttribute("data-lang")))
				|| block.getAttribute("data-name")
				|| block.getAttribute("data-lang");
			title.innerHTML = lang;

			let icon = document.createElement("i");
			icon.classList.add("icon");

			let button = document.createElement("button");
			let copyCodeText = document.getElementById("copy-code-text").textContent;
			button.setAttribute("title", copyCodeText);
			button.appendChild(icon);

			let header = document.createElement("div");
			header.classList.add("header");
			header.appendChild(title);
			header.appendChild(button);

			let container = document.createElement("div");
			container.classList.add("pre-container");
			container.appendChild(header);

			block.parentNode.insertBefore(container, block);
			container.appendChild(block);

			button.addEventListener("click", async () => {
				await copyCode(block, header, button);
			});
		}
	});

	async function copyCode(block, header, button) {
		let code = block.querySelector("code");
		let text = code.innerText;

		await navigator.clipboard.writeText(text);

		header.classList.add("active");
		button.setAttribute("disabled", true);

		header.addEventListener("animationend", () => {
			header.classList.remove("active");
			button.removeAttribute("disabled");
		}, { once: true });
	}
});
