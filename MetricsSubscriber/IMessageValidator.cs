namespace MetricsSubscriber;

public class ValidationException(string? message, Exception? innerException = default) : Exception(message, innerException)
{
}

public interface IMessageValidator<TNotification>
{
	bool Validate(ref readonly TNotification? notification);
}

public class DefaultValidator<TNotification> : IMessageValidator<TNotification>
{
	public bool Validate(ref readonly TNotification? notification)
	{
		return notification switch
		{
			null => throw new ValidationException("Notification of type T was null"),
			_ => true,
		};
	}
}
