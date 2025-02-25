namespace MetricsSubscriber;

public interface IMessageParser<out TNotification>
{
	TNotification? TryParse(string message)
	{
		return default;
	}
}

public class DefaultParser<TNotification> : IMessageParser<TNotification>
{
	public TNotification? TryParse(string message) => default;
}

