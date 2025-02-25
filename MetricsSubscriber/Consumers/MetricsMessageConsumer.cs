using System.Text.Json;
using MassTransit;
using MediatR;
using MetricsSubscriber.Entity;
using MetricsSubscriber.Models.Bus;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.Logging;

namespace MetricsSubscriber.Consumers;

public class MetricsConsumer(
		IMediator mediator,
		IMessageValidator<MetricsMessage> validator,
		IMessageTransformer<MetricsMessage, Metrics> transformer,
		ILogger<MetricsConsumer> logger) : IConsumer<MetricsMessage>
{
	public async Task Consume(ConsumeContext<MetricsMessage> context)
	{
		var message = context.Message;
		logger.LogInformation("Received message: {Message}", JsonSerializer.Serialize(message));

		if (!validator.Validate(in message))
		{
			logger.LogWarning("Message validation failed: {Message}", JsonSerializer.Serialize(message));
			throw new ValidationException("Message validation failed");
		}

		var transformed = transformer.Transform(message);

		var notification = new MetricsNotification(
			transformed.Epoch,
			transformed.Loss,
			transformed.Accuracy,
			transformed.MeanPPV,
			transformed.MeanFPR,
			transformed.MeanRecall);

		await mediator.Publish(notification, context.CancellationToken);
	}
}
