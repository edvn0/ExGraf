

using System.Text.Json;
using MassTransit;
using MediatR;
using MetricsSubscriber.Models.Bus;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.Logging;

namespace MetricsSubscriber.Consumers;

public class ModelConfigurationConsumer(
	IMediator mediator,
	ILogger<ModelConfigurationConsumer> logger) : IConsumer<ModelConfigurationMessage>
{
	public async Task Consume(ConsumeContext<ModelConfigurationMessage> context)
	{
		var message = context.Message;
		logger.LogInformation("Received message: {Message}", JsonSerializer.Serialize(message));

		var notification = new ModelConfigurationNotification(
			message.Name,
			[.. message.Layers],
			message.LearningRate);

		await mediator.Publish(notification, context.CancellationToken);
	}
}
