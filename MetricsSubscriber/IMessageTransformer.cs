using MediatR;

namespace MetricsSubscriber;

public interface IMessageTransformer<TNotification> where TNotification : INotification
{
	void Transform(ref TNotification notification);
}


public class DefaultTransformer<TNotification> : IMessageTransformer<TNotification> where TNotification : INotification
{
	public void Transform(ref TNotification notification) { }
}


