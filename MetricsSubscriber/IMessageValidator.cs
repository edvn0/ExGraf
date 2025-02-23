using MediatR;

namespace MetricsSubscriber;

public interface IMessageValidator<TNotification> where TNotification : INotification
{
	bool Validate(ref readonly TNotification notification)
	{
		return true;
	}
}

public class DefaultValidator<TNotification> : IMessageValidator<TNotification> where TNotification : INotification
{
	public bool Validate(ref readonly TNotification notification) => true;
}
