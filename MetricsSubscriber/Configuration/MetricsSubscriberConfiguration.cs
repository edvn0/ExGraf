
using MediatR;
using Microsoft.Extensions.DependencyInjection;

namespace MetricsSubscriber.Configuration;

public static class ServiceCollectionExtensions
{
	public static void AddMessageHandlingFor<T, TParse, TValidate, TTransform>(this IServiceCollection services)
		where T : INotification
		where TParse : class, IMessageParser<T>
		where TValidate : class, IMessageValidator<T>
		where TTransform : class, IMessageTransformer<T>
	{
		services.AddTransient<IMessageParser<T>, TParse>();
		services.AddTransient<IMessageValidator<T>, TValidate>();
		services.AddTransient<IMessageTransformer<T>, TTransform>();
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
