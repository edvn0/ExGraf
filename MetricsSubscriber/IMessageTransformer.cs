using MediatR;

namespace MetricsSubscriber;

public interface IMessageTransformer<TNotification>
{
	void Transform(ref TNotification notification);
}


public class DefaultTransformer<TNotification> : IMessageTransformer<TNotification>
{
	public void Transform(ref TNotification notification) { }
}


