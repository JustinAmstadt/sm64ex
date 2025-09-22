import asyncio
import websockets

from buttons import ControllerButton

async def hello():
    uri = "ws://localhost:8080"  # Replace with your server address
    async with websockets.connect(uri, subprotocols=[]) as websocket:
        # Send a message
        for _ in range(10):
            await websocket.send(str(int(ControllerButton.A_BUTTON)))
        print("Message sent to server.")


# Run the client
asyncio.run(hello())
