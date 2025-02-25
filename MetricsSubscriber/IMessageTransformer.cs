using MediatR;

namespace MetricsSubscriber;

public interface IMessageTransformer<in TNotification, out TOutput>
{
	TOutput Transform(TNotification notification);
}


public class DefaultTransformer<TNotification> : IMessageTransformer<TNotification, TNotification>
{
	public TNotification Transform(TNotification notification) { return notification; }
}


