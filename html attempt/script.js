async function submitMessage() {
    let userInput = document.getElementById("userInput").value;
    document.getElementById("chatbox").innerHTML += `<p>User: ${userInput}</p>`;
    
    // Send user input to backend server for processing
    let response = await fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput })
    });
    
    let data = await response.json();
    document.getElementById("chatbox").innerHTML += `<p>Bot: ${data.response}</p>`;
}
