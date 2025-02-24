using System.Reflection;
using MassTransit;
using MediatR;
using MetricsSubscriber;
using MetricsSubscriber.Configuration;
using MetricsSubscriber.Implementations;
using MetricsSubscriber.Models.Vertical;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var host = Host.CreateDefaultBuilder()
	.ConfigureServices((context, services) =>
	{
		services.AddMediatR(cfg => cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly()));
		services.AddTransient<ISubscriberSocket, NetMqSubscriberSocket>();

		services.AddMessageHandlingFor<MetricsUpdatedEvent, MetricsMessageParser>();
		services.AddMetricsMassTransit();

		services.AddLogging();
	})
	.Build();

try
{
	await host.StartAsync();
	await host.WaitForShutdownAsync();
}
finally
{
	if (host is IAsyncDisposable disposableHost)
		await disposableHost.DisposeAsync();
	else
		host.Dispose();
}

