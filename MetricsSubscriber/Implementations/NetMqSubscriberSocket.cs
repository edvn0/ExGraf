using NetMQ;
using NetMQ.Sockets;

namespace MetricsSubscriber.Implementations;

public class NetMqSubscriberSocket : ISubscriberSocket
{
	private readonly SubscriberSocket _socket;

	public NetMqSubscriberSocket()
	{
		_socket = new SubscriberSocket();
		_socket.Connect("tcp://localhost:5555");
		_socket.Subscribe("");
	}

	public bool TryReceiveMessage(TimeSpan timeout, out string message)
	{
		message = string.Empty;

		// NetMQ's TryReceive with timeout
		if (!_socket.TryReceiveFrameString(timeout, out string? frame))
		{
			return false;
		}

		message = frame;
		return true;
	}

	public void Dispose()
	{
		_socket.Dispose();
	}
}
