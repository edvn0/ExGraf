using MediatR;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace MetricsSubscriber;

public class MessageSubscriberBackgroundService<TNotification>(
	IMediator mediator,
	ISubscriberSocket socket,
	IMessageParser<TNotification> parser,
	IMessageValidator<TNotification> validator,
	IMessageTransformer<TNotification> transformer,
	ILogger<MessageSubscriberBackgroundService<TNotification>> logger) : BackgroundService
	where TNotification : INotification
{
	protected override async Task ExecuteAsync(CancellationToken stoppingToken)
	{
		try
		{
			while (!stoppingToken.IsCancellationRequested)
			{
				try
				{
					if (!socket.TryReceiveMessage(TimeSpan.FromMilliseconds(100), out string message)) continue;

					var maybeNotification = parser.TryParse(message);
					if (maybeNotification is not null)
					{
						if (!validator.Validate(ref maybeNotification))
						{
							continue;
						}

						transformer.Transform(ref maybeNotification!);
						await mediator.Publish(maybeNotification, stoppingToken);
					}
					else
					{
						logger.LogWarning("Failed to parse message: {Message}", message);
					}
				}
				catch (OperationCanceledException)
				{
					break;
				}
				catch (Exception ex)
				{
					logger.LogError(ex, "Could not receive message");
				}
			}
		}
		finally
		{
			logger.LogInformation("Shutting down '{Name}'.",
				nameof(MessageSubscriberBackgroundService<TNotification>));
			socket.Dispose();
		}
	}
}
