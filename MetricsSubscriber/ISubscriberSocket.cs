namespace MetricsSubscriber;

public interface ISubscriberSocket : IDisposable
{
	bool TryReceiveMessage(TimeSpan timeout, out string message);
}
