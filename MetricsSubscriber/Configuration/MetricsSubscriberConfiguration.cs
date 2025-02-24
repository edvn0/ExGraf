
using System.Reflection;
using System.Text.Json;
using MassTransit;
using MediatR;
using MetricsSubscriber.Entity;
using MetricsSubscriber.Models.Bus;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace MetricsSubscriber.Configuration;

public static class ServiceCollectionExtensions
{
	public class MetricsValidator : IMessageValidator<MetricsMessage>
	{
		public bool Validate(ref readonly MetricsMessage? message)
		{
			return message switch
			{
				not null => message.Epoch >= 0
				&& message.Loss >= 0
				&& message.Accuracy >= 0
				&& message.Accuracy <= 100
				&& message.MeanPPV >= 0
				&& message.MeanPPV <= 1
				&& message.MeanFPR >= 0
				&& message.MeanFPR <= 1
				&& message.MeanRecall >= 0
				&& message.MeanRecall <= 1,
				_ => true,
			};

		}
	}

	public class MetricsTransformer : IMessageTransformer<MetricsMessage, Metrics>
	{
		public Metrics Transform(MetricsMessage message)
		{
			var accuracy = message.Accuracy;
			if (accuracy > 1)
			{
				accuracy /= 100.0;
			}

			return new()
			{
				Epoch = message.Epoch,
				Loss = message.Loss,
				Accuracy = accuracy,
				MeanPPV = message.MeanPPV,
				MeanFPR = message.MeanFPR,
				MeanRecall = message.MeanRecall,
				ModelConfiguration = null,
			};
		}
	}

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

	public static void AddMassTransit(this IServiceCollection services)
	{
		services.AddSingleton<IMessageValidator<MetricsMessage>, MetricsValidator>();
		services.AddSingleton<IMessageTransformer<MetricsMessage, Metrics>, MetricsTransformer>();

		services.AddMassTransit(x =>
		{
			// Get all consumers from the assembly
			x.AddConsumers(Assembly.GetExecutingAssembly());

			x.UsingRabbitMq((context, cfg) =>
			{
				var configuration = context.GetRequiredService<IConfiguration>();

				var section = configuration.GetSection("RabbitMq")
					?? throw new InvalidOperationException("RabbitMq section not found in configuration");
				var host = section.GetValue<string>("Host")
					?? throw new InvalidOperationException("RabbitMq Host not found in configuration");
				var username = section.GetValue<string>("Username")
					?? throw new InvalidOperationException("RabbitMq Username not found in configuration");
				var password = section.GetValue<string>("Password")
					?? throw new InvalidOperationException("RabbitMq Password not found in configuration");
				var virtualHost = section.GetValue<string>("VirtualHost") ?? "/";
				var timeSpan = section.GetValue<TimeSpan?>("RetryInterval") ?? TimeSpan.FromSeconds(5);

				cfg.Host(host, virtualHost, h =>
				{
					h.Username(username);
					h.Password(password);
				});

				cfg.ReceiveEndpoint("metrics", e =>
				{
					e.UseRawJsonDeserializer(isDefault: true);
					e.UseRawJsonSerializer(isDefault: true);

					e.ConfigureConsumer<MetricsConsumer>(context);
					e.UseMessageRetry(r => r.Intervals(100, 500, 1000));
					e.UseDelayedRedelivery(r => r.Intervals(timeSpan));
				});
			});
		});
	}
}

