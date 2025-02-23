using MediatR;

namespace MetricsSubscriber;

public interface IMessageParser<out TNotification> where TNotification : INotification
{
	TNotification? TryParse(string message)
	{
		return default;
	}
}

public class DefaultParser<TNotification> : IMessageParser<TNotification> where TNotification : INotification
{
	public TNotification? TryParse(string message) => default;
}

