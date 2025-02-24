
using System.Net.Mime;
using System.Text;
using System.Text.Json;
using MassTransit;
using MassTransit.Context;
using MediatR;
using MetricsSubscriber.Models.Bus;
using MetricsSubscriber.Models.Vertical;
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

	public class MetricsTransformer : IMessageTransformer<MetricsMessage>
	{
		public void Transform(ref MetricsMessage message)
		{
			var accuracy = message.Accuracy;
			if (accuracy > 1)
			{
				accuracy /= 100.0;
			}

			message = message with { Accuracy = accuracy };
		}
	}

	public class MetricsConsumer(
		IMediator mediator,
		IMessageValidator<MetricsMessage> validator,
		IMessageTransformer<MetricsMessage> transformer,
		ILogger<MetricsConsumer> logger) : IConsumer<MetricsMessage>
	{
		public async Task Consume(ConsumeContext<MetricsMessage> context)
		{
			var message = context.Message;

			if (!validator.Validate(in message))
			{
				logger.LogWarning("Message validation failed: {Message}", JsonSerializer.Serialize(message));
				throw new ValidationException("Message validation failed");
			}

			transformer.Transform(ref message);

			var notification = new MetricsNotification(
				message.Epoch,
				message.Loss,
				message.Accuracy,
				message.MeanPPV,
				message.MeanFPR,
				message.MeanRecall);

			await mediator.Publish(notification, context.CancellationToken);
		}
	}



	public static void AddMetricsMassTransit(this IServiceCollection services)
	{
		services.AddSingleton<IMessageValidator<MetricsMessage>, MetricsValidator>();
		services.AddSingleton<IMessageTransformer<MetricsMessage>, MetricsTransformer>();

		services.AddMassTransit(x =>
		{
			x.AddConsumer<MetricsConsumer>();

			x.UsingRabbitMq((context, cfg) =>
			{
				cfg.Host("localhost", "/", h =>
				{
					h.Username("guest");
					h.Password("guest");
				});

				cfg.ReceiveEndpoint("metrics_queue", e =>
				{
					e.ConfigureConsumer<MetricsConsumer>(context);

					e.UseMessageRetry(r =>
					{
						r.Intervals(100, 500, 1000);
					});

					e.UseDelayedRedelivery(r =>
					{
						r.Intervals(TimeSpan.FromMinutes(5));
					});
				});
			});
		});
	}

	public static void AddMessageHandlingFor<T, TParse, TValidate, TTransform>(this IServiceCollection services)
		where T : INotification
		where TParse : class, IMessageParser<T>
		where TValidate : class, IMessageValidator<T>
		where TTransform : class, IMessageTransformer<T>
	{
		services.AddSingleton<IMessageParser<T>, TParse>();
		services.AddSingleton<IMessageValidator<T>, TValidate>();
		services.AddSingleton<IMessageTransformer<T>, TTransform>();
		services.AddHostedService<MessageSubscriberBackgroundService<T>>();
	}

	public static void AddMessageHandlingFor<T, TParse>(this IServiceCollection services)
		where T : INotification
		where TParse : class, IMessageParser<T>
	{
		services.AddMessageHandlingFor<T, TParse, DefaultValidator<T>, DefaultTransformer<T>>();
	}

	public static void AddMessageHandlingFor<T>(this IServiceCollection services)
		where T : INotification
	{
		services.AddMessageHandlingFor<T, DefaultParser<T>, DefaultValidator<T>, DefaultTransformer<T>>();
	}
}

